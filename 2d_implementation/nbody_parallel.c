#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#define G 6.67430e-11  // Gravitational constant
#define EPSILON 1e-9   // Softening parameter to avoid divisions by zero

typedef struct {
    int id;
    double mass;
    double pos_x, pos_y;
    double vel_x, vel_y;
    double prev_pos_x, prev_pos_y;  // Previous positions for Verlet
    double force_x, force_y;
    int first_step;  // Flag to indicate if it's the first step
} Body;

// PHASE 1: PARTITIONING - Force partitioning
// Partitions the force calculation to reduce communications
void partition_force_calculation(int total_n, int rank, int size, 
                                int* start_i, int* end_i, int* local_pairs) {
    // Upper triangle distribution of the interaction matrix
    long total_pairs = (long)total_n * (total_n - 1) / 2;
    long pairs_per_process = total_pairs / size;
    long remainder = total_pairs % size;
    
    long my_start_pair = rank * pairs_per_process + (rank < remainder ? rank : remainder);
    long my_end_pair = my_start_pair + pairs_per_process + (rank < remainder ? 1 : 0);
    
    // Convert the index of the pair to the indices i,j
    long pair_count = 0;
    *start_i = 0;
    *end_i = total_n;
    *local_pairs = (int)(my_end_pair - my_start_pair);
    
    // Find the beginning indexes for this process
    for (int i = 0; i < total_n && pair_count < my_start_pair; i++) {
        long pairs_in_row = total_n - i - 1;
        if (pair_count + pairs_in_row <= my_start_pair) {
            pair_count += pairs_in_row;
            *start_i = i + 1;
        } else {
            break;
        }
    }
}

// PHASE 2: COMMUNICATION - Data structure to reduce communications (2D version)
typedef struct {
    double pos_x, pos_y;
    double mass;
} BodyPos;

typedef struct {
    double force_x, force_y;
    int body_id;
} BodyForce;

// Function to read data (root process only, 2D version)
int read_csv(const char* filename, Body** bodies) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file %s\n", filename);
        return -1;
    }
    
    char line[1024];
    int n = 0;
    
    // Count rows
    fgets(line, sizeof(line), file);
    while (fgets(line, sizeof(line), file)) {
        n++;
    }
    
    *bodies = (Body*)malloc(n * sizeof(Body));
    if (!*bodies) {
        printf("Memory allocation error\n");
        fclose(file);
        return -1;
    }
    
    rewind(file);
    fgets(line, sizeof(line), file);
    
    for (int i = 0; i < n; i++) {
        if (fgets(line, sizeof(line), file)) {
            sscanf(line, "%d,%lf,%lf,%lf,%lf,%lf",
                   &(*bodies)[i].id,
                   &(*bodies)[i].mass,
                   &(*bodies)[i].pos_x, &(*bodies)[i].pos_y,
                   &(*bodies)[i].vel_x, &(*bodies)[i].vel_y);
            
            // Initialize forces to zero
            (*bodies)[i].force_x = 0.0;
            (*bodies)[i].force_y = 0.0;
            
            // Initialize previous positions (not used in first step)
            (*bodies)[i].prev_pos_x = 0.0;
            (*bodies)[i].prev_pos_y = 0.0;
            (*bodies)[i].first_step = 1;
        }
    }
    
    fclose(file);
    return n;
}

// PHASE 3: AGGLOMERATION - Force calculation with less communication (2D version)
void calculate_forces(Body* bodies, int total_n, int rank, int size,
                      double* local_forces_x, double* local_forces_y) {
    
    // Reset local forces
    for (int i = 0; i < total_n; i++) {
        local_forces_x[i] = 0.0;
        local_forces_y[i] = 0.0;
    }
    
    // Each process computes a subset of the interactions
    for (int i = rank; i < total_n; i += size) {
        for (int j = i + 1; j < total_n; j++) {
            // Compute the distance
            double dx = bodies[j].pos_x - bodies[i].pos_x;
            double dy = bodies[j].pos_y - bodies[i].pos_y;
            
            double dist_squared = dx*dx + dy*dy + EPSILON*EPSILON;
            double dist = sqrt(dist_squared);
            double dist_cubed = dist_squared * dist;
            
            // Compute force
            double force_magnitude = G * bodies[i].mass * bodies[j].mass / dist_cubed;
            
            double force_x = force_magnitude * dx;
            double force_y = force_magnitude * dy;
            
            // Application of Newton's third law
            local_forces_x[i] += force_x;
            local_forces_y[i] += force_y;
            
            local_forces_x[j] -= force_x;
            local_forces_y[j] -= force_y;
        }
    }
}

// PHASE 4: MAPPING - Efficient force reduction (2D version)
void reduce_forces(double* local_forces_x, double* local_forces_y,
                   double* global_forces_x, double* global_forces_y,
                   int total_n) {
    
    // Use MPI_Allreduce to sum the forces of all processes
    MPI_Allreduce(local_forces_x, global_forces_x, total_n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_forces_y, global_forces_y, total_n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

// Update positions and velocities using Verlet integration (2D version)
void update_bodies_verlet(Body* bodies, int total_n, 
                         double* global_forces_x, double* global_forces_y,
                         double dt) {
    for (int i = 0; i < total_n; i++) {
        if (bodies[i].first_step) {
            // For the first step, use Euler to calculate the previous position
            double acc_x = global_forces_x[i] / bodies[i].mass;
            double acc_y = global_forces_y[i] / bodies[i].mass;
            
            // Save current position as previous
            bodies[i].prev_pos_x = bodies[i].pos_x;
            bodies[i].prev_pos_y = bodies[i].pos_y;
            
            // Calculate new position using modified Euler
            bodies[i].pos_x += bodies[i].vel_x * dt + 0.5 * acc_x * dt * dt;
            bodies[i].pos_y += bodies[i].vel_y * dt + 0.5 * acc_y * dt * dt;
            
            // Update velocity
            bodies[i].vel_x += acc_x * dt;
            bodies[i].vel_y += acc_y * dt;
            
            bodies[i].first_step = 0;
        } else {
            // Use Verlet integration for subsequent steps
            double acc_x = global_forces_x[i] / bodies[i].mass;
            double acc_y = global_forces_y[i] / bodies[i].mass;
            
            // Save current position
            double temp_pos_x = bodies[i].pos_x;
            double temp_pos_y = bodies[i].pos_y;
            
            // Calculate new position using Verlet
            bodies[i].pos_x = 2.0 * bodies[i].pos_x - bodies[i].prev_pos_x + acc_x * dt * dt;
            bodies[i].pos_y = 2.0 * bodies[i].pos_y - bodies[i].prev_pos_y + acc_y * dt * dt;
            
            // Update previous position
            bodies[i].prev_pos_x = temp_pos_x;
            bodies[i].prev_pos_y = temp_pos_y;
            
            // Calculate velocity using position difference
            bodies[i].vel_x = (bodies[i].pos_x - bodies[i].prev_pos_x) / (2.0 * dt);
            bodies[i].vel_y = (bodies[i].pos_y - bodies[i].prev_pos_y) / (2.0 * dt);
        }
        
        // Update forces in the body structure
        bodies[i].force_x = global_forces_x[i];
        bodies[i].force_y = global_forces_y[i];
    }
}

void print_bodies(Body* bodies, int n, int rank) {
    if (rank == 0) {
        printf("ID\tMass\t\tPos_X\t\tPos_Y\t\tVel_X\t\tVel_Y\n");
        printf("---------------------------------------------------------------\n");
        for (int i = 0; i < n; i++) {
            printf("%d\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\n",
                   bodies[i].id, bodies[i].mass,
                   bodies[i].pos_x, bodies[i].pos_y,
                   bodies[i].vel_x, bodies[i].vel_y);
        }
        printf("\n");
    }
}

void save_results(const char* filename, Body* bodies, int n, int rank) {
    if (rank == 0) {
        FILE* file = fopen(filename, "w");
        if (!file) {
            printf("Error opening output file %s\n", filename);
            return;
        }
        
        fprintf(file, "id,mass,pos_x,pos_y,vel_x,vel_y\n");
        for (int i = 0; i < n; i++) {
            fprintf(file, "%d,%.15f,%.15f,%.15f,%.15f,%.15f\n",
                    bodies[i].id, bodies[i].mass,
                    bodies[i].pos_x, bodies[i].pos_y,
                    bodies[i].vel_x, bodies[i].vel_y);
        }
        
        fclose(file);
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc != 2) {
        if (rank == 0) {
            printf("Usage: mpirun -np <num_proc> %s <csv_filename>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    Body* bodies = NULL;
    int total_n = 0;
    
    // Only root reads the file
    if (rank == 0) {
        total_n = read_csv(argv[1], &bodies);
        if (total_n < 0) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        printf("Read %d bodies from file %s\n", total_n, argv[1]);
        printf("Using %d MPI processes\n", size);
        printf("Using Verlet integration method\n\n");
    }
    
    // Broadcast of the number of bodies
    MPI_Bcast(&total_n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // All processes allocate memory for all bodies
    if (rank != 0) {
        bodies = (Body*)malloc(total_n * sizeof(Body));
    }
    
    // Broadcast of all body data
    MPI_Bcast(bodies, total_n * sizeof(Body), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    // Arrays allocation for forces (2D version)
    double* local_forces_x = (double*)calloc(total_n, sizeof(double));
    double* local_forces_y = (double*)calloc(total_n, sizeof(double));
    double* global_forces_x = (double*)calloc(total_n, sizeof(double));
    double* global_forces_y = (double*)calloc(total_n, sizeof(double));
    
    // Simulation parameters
    double dt = 0.01;
    int steps = 100;
    
    if (rank == 0) {
        printf("Initial state:\n");
        print_bodies(bodies, total_n, rank);
    }
    
    // Runtime measurement
    MPI_Barrier(MPI_COMM_WORLD);  // Synchronizes all processes
    double start_time = MPI_Wtime();
    
    for (int step = 0; step < steps; step++) {
        // Parallel calculation of forces
        calculate_forces(bodies, total_n, rank, size,
                        local_forces_x, local_forces_y);
        
        // Reduction of forces
        reduce_forces(local_forces_x, local_forces_y,
                     global_forces_x, global_forces_y, total_n);
        
        // Synchronized updating of positions and velocities using Verlet
        update_bodies_verlet(bodies, total_n, 
                           global_forces_x, global_forces_y, dt);
        
        // Periodic print
        if (step % 20 == 0 && rank == 0) {
            printf("Step %d:\n", step);
            print_bodies(bodies, total_n, rank);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);  // Synchronize before measuring the final time
    double end_time = MPI_Wtime();
    double execution_time = end_time - start_time;
    
    if (rank == 0) {
        printf("Execution time for parallel: %.6f seconds\n", execution_time);
        printf("Time per force calculation step: %.6f seconds\n", execution_time / steps);
        
        // Save results
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "output_parallel_%s", argv[1]);
        save_results(output_filename, bodies, total_n, rank);
        printf("Results saved to %s\n", output_filename);
    }
    
    // Cleanup (2D version)
    free(bodies);
    free(local_forces_x);
    free(local_forces_y);
    free(global_forces_x);
    free(global_forces_y);
    
    MPI_Finalize();
    return 0;
}