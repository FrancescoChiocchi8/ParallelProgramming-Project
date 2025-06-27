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
    double pos_x, pos_y, pos_z;
    double vel_x, vel_y, vel_z;
    double force_x, force_y, force_z;
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

// PHASE 2: COMMUNICATION - Data structure to reduce communications
typedef struct {
    double pos_x, pos_y, pos_z;
    double mass;
} BodyPos;

typedef struct {
    double force_x, force_y, force_z;
    int body_id;
} BodyForce;

// Function to read data (root process only)
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
            sscanf(line, "%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf",
                   &(*bodies)[i].id,
                   &(*bodies)[i].mass,
                   &(*bodies)[i].pos_x, &(*bodies)[i].pos_y, &(*bodies)[i].pos_z,
                   &(*bodies)[i].vel_x, &(*bodies)[i].vel_y, &(*bodies)[i].vel_z);
            
            (*bodies)[i].force_x = 0.0;
            (*bodies)[i].force_y = 0.0;
            (*bodies)[i].force_z = 0.0;
        }
    }
    
    fclose(file);
    return n;
}

// PHASE 3: AGGLOMERATION - Force calculation with less communication
void calculate_forces(Body* bodies, int total_n, int rank, int size,
                               double* local_forces_x, double* local_forces_y, double* local_forces_z) {
    
    // Reset local forces
    for (int i = 0; i < total_n; i++) {
        local_forces_x[i] = 0.0;
        local_forces_y[i] = 0.0;
        local_forces_z[i] = 0.0;
    }
    
    // Each process computes a subset of the interactions
    for (int i = rank; i < total_n; i += size) {
        for (int j = i + 1; j < total_n; j++) {
            // Compute the distance
            double dx = bodies[j].pos_x - bodies[i].pos_x;
            double dy = bodies[j].pos_y - bodies[i].pos_y;
            double dz = bodies[j].pos_z - bodies[i].pos_z;
            
            double dist_squared = dx*dx + dy*dy + dz*dz + EPSILON*EPSILON;
            double dist = sqrt(dist_squared);
            double dist_cubed = dist_squared * dist;
            
            // Compute force
            double force_magnitude = G * bodies[i].mass * bodies[j].mass / dist_cubed;
            
            double force_x = force_magnitude * dx;
            double force_y = force_magnitude * dy;
            double force_z = force_magnitude * dz;
            
            // Application of Newton's third law
            local_forces_x[i] += force_x;
            local_forces_y[i] += force_y;
            local_forces_z[i] += force_z;
            
            local_forces_x[j] -= force_x;
            local_forces_y[j] -= force_y;
            local_forces_z[j] -= force_z;
        }
    }
}

// PHASE 4: MAPPING - Efficient force reduction
void reduce_forces(double* local_forces_x, double* local_forces_y, double* local_forces_z,
                   double* global_forces_x, double* global_forces_y, double* global_forces_z,
                   int total_n) {
    
    // Use MPI_Allreduce to sum the forces of all processes
    MPI_Allreduce(local_forces_x, global_forces_x, total_n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_forces_y, global_forces_y, total_n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_forces_z, global_forces_z, total_n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

// Update positions and speed (each process updates all bodies)
void update_bodies_synchronized(Body* bodies, int total_n, 
                               double* global_forces_x, double* global_forces_y, double* global_forces_z,
                               double dt) {
    for (int i = 0; i < total_n; i++) {
        // Update velocity
        bodies[i].vel_x += (global_forces_x[i] / bodies[i].mass) * dt;
        bodies[i].vel_y += (global_forces_y[i] / bodies[i].mass) * dt;
        bodies[i].vel_z += (global_forces_z[i] / bodies[i].mass) * dt;
        
        // Update position
        bodies[i].pos_x += bodies[i].vel_x * dt;
        bodies[i].pos_y += bodies[i].vel_y * dt;
        bodies[i].pos_z += bodies[i].vel_z * dt;
        
        // Upgrade the forces in the body structure
        bodies[i].force_x = global_forces_x[i];
        bodies[i].force_y = global_forces_y[i];
        bodies[i].force_z = global_forces_z[i];
    }
}

// Synchronized broadcast of positions
void synchronize_positions(Body* bodies, int total_n, int rank) {
    // Each process already has all positions updated identically
    // This eliminates the need for additional communication
}

void print_bodies(Body* bodies, int n, int rank) {
    if (rank == 0) {
        printf("ID\tMass\t\tPos_X\t\tPos_Y\t\tPos_Z\t\tVel_X\t\tVel_Y\t\tVel_Z\n");
        printf("-----------------------------------------------------------------------------------------\n");
        for (int i = 0; i < n; i++) {
            printf("%d\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\n",
                   bodies[i].id, bodies[i].mass,
                   bodies[i].pos_x, bodies[i].pos_y, bodies[i].pos_z,
                   bodies[i].vel_x, bodies[i].vel_y, bodies[i].vel_z);
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
        
        fprintf(file, "id,mass,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z\n");
        for (int i = 0; i < n; i++) {
            fprintf(file, "%d,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f\n",
                    bodies[i].id, bodies[i].mass,
                    bodies[i].pos_x, bodies[i].pos_y, bodies[i].pos_z,
                    bodies[i].vel_x, bodies[i].vel_y, bodies[i].vel_z);
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
        printf("Using %d MPI processes\n\n", size);
    }
    
    // Broadcast of the number of bodies
    MPI_Bcast(&total_n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // All processes allocate memory for all bodies
    if (rank != 0) {
        bodies = (Body*)malloc(total_n * sizeof(Body));
    }
    
    // Broadcast of all body data
    MPI_Bcast(bodies, total_n * sizeof(Body), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    // Arrays allocation for forces
    double* local_forces_x = (double*)calloc(total_n, sizeof(double));
    double* local_forces_y = (double*)calloc(total_n, sizeof(double));
    double* local_forces_z = (double*)calloc(total_n, sizeof(double));
    double* global_forces_x = (double*)calloc(total_n, sizeof(double));
    double* global_forces_y = (double*)calloc(total_n, sizeof(double));
    double* global_forces_z = (double*)calloc(total_n, sizeof(double));
    
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
                                 local_forces_x, local_forces_y, local_forces_z);
        
        // Reduction of forces
        reduce_forces(local_forces_x, local_forces_y, local_forces_z,
                     global_forces_x, global_forces_y, global_forces_z, total_n);
        
        // Synchronized updating of positions and speeds
        update_bodies_synchronized(bodies, total_n, 
                                 global_forces_x, global_forces_y, global_forces_z, dt);
        
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
    
    // Cleanup
    free(bodies);
    free(local_forces_x);
    free(local_forces_y);
    free(local_forces_z);
    free(global_forces_x);
    free(global_forces_y);
    free(global_forces_z);
    
    MPI_Finalize();
    return 0;
}