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
    double prev_pos_x, prev_pos_y;  // Previous positions for Verlet integration
    double force_x, force_y;
    int first_step;  // Flag to indicate if it's the first integration step
} Body;

// Read body data from CSV file (only root process)
int read_csv(const char* filename, Body** bodies) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file %s\n", filename);
        return -1;
    }
    
    char line[1024];
    int n = 0;
    
    // Count the number of bodies
    fgets(line, sizeof(line), file);  // Skip header
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
    fgets(line, sizeof(line), file);  // Skip header again
    
    for (int i = 0; i < n; i++) {
        if (fgets(line, sizeof(line), file)) {
            sscanf(line, "%d,%lf,%lf,%lf,%lf,%lf",
                   &(*bodies)[i].id,
                   &(*bodies)[i].mass,
                   &(*bodies)[i].pos_x, &(*bodies)[i].pos_y,
                   &(*bodies)[i].vel_x, &(*bodies)[i].vel_y);
            
            // Initialize forces and previous positions
            (*bodies)[i].force_x = 0.0;
            (*bodies)[i].force_y = 0.0;
            (*bodies)[i].prev_pos_x = 0.0;
            (*bodies)[i].prev_pos_y = 0.0;
            (*bodies)[i].first_step = 1;
        }
    }
    
    fclose(file);
    return n;
}

// Calculate gravitational forces between all bodies (parallel computation)
void calculate_forces(Body* bodies, int total_n, int rank, int size,
                      double* local_forces_x, double* local_forces_y) {
    
    // Reset local force arrays
    for (int i = 0; i < total_n; i++) {
        local_forces_x[i] = 0.0;
        local_forces_y[i] = 0.0;
    }
    
    // Each process computes a subset of the force interactions
    for (int i = rank; i < total_n; i += size) {
        for (int j = i + 1; j < total_n; j++) {
            // Calculate distance between bodies i and j
            double dx = bodies[j].pos_x - bodies[i].pos_x;
            double dy = bodies[j].pos_y - bodies[i].pos_y;
            
            double dist_squared = dx*dx + dy*dy + EPSILON*EPSILON;
            double dist = sqrt(dist_squared);
            double dist_cubed = dist_squared * dist;
            
            // Calculate gravitational force magnitude
            double force_magnitude = G * bodies[i].mass * bodies[j].mass / dist_cubed;
            
            double force_x = force_magnitude * dx;
            double force_y = force_magnitude * dy;
            
            // Apply Newton's third law: F_ij = -F_ji
            local_forces_x[i] += force_x;
            local_forces_y[i] += force_y;
            
            local_forces_x[j] -= force_x;
            local_forces_y[j] -= force_y;
        }
    }
}

// Reduce forces from all processes using MPI_Allreduce
void reduce_forces(double* local_forces_x, double* local_forces_y,
                   double* global_forces_x, double* global_forces_y,
                   int total_n) {
    
    MPI_Allreduce(local_forces_x, global_forces_x, total_n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_forces_y, global_forces_y, total_n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

// Update positions and velocities using Verlet integration
void update_bodies_verlet(Body* bodies, int total_n, 
                         double* global_forces_x, double* global_forces_y,
                         double dt) {
    for (int i = 0; i < total_n; i++) {
        if (bodies[i].first_step) {
            // First step: use modified Euler method to establish previous position
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
            // Subsequent steps: use Verlet integration
            double acc_x = global_forces_x[i] / bodies[i].mass;
            double acc_y = global_forces_y[i] / bodies[i].mass;
            
            // Save current position
            double temp_pos_x = bodies[i].pos_x;
            double temp_pos_y = bodies[i].pos_y;
            
            // Calculate new position using Verlet formula
            bodies[i].pos_x = 2.0 * bodies[i].pos_x - bodies[i].prev_pos_x + acc_x * dt * dt;
            bodies[i].pos_y = 2.0 * bodies[i].pos_y - bodies[i].prev_pos_y + acc_y * dt * dt;
            
            // Update previous position
            bodies[i].prev_pos_x = temp_pos_x;
            bodies[i].prev_pos_y = temp_pos_y;
            
            // Calculate velocity using centered difference
            bodies[i].vel_x = (bodies[i].pos_x - bodies[i].prev_pos_x) / (2.0 * dt);
            bodies[i].vel_y = (bodies[i].pos_y - bodies[i].prev_pos_y) / (2.0 * dt);
        }
        
        // Update forces in the body structure
        bodies[i].force_x = global_forces_x[i];
        bodies[i].force_y = global_forces_y[i];
    }
}

// Print body information (only root process)
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

// Save simulation results to CSV file (only root process)
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
    
    // Only root process reads the input file
    if (rank == 0) {
        total_n = read_csv(argv[1], &bodies);
        if (total_n < 0) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        printf("Read %d bodies from file %s\n", total_n, argv[1]);
        printf("Using %d MPI processes\n", size);
        printf("Using Verlet integration method\n\n");
    }
    
    // Broadcast number of bodies to all processes
    MPI_Bcast(&total_n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // All non-root processes allocate memory for bodies
    if (rank != 0) {
        bodies = (Body*)malloc(total_n * sizeof(Body));
    }
    
    // Broadcast all body data to all processes
    MPI_Bcast(bodies, total_n * sizeof(Body), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    // Allocate arrays for force calculations
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
    
    // Synchronize all processes before starting simulation
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    
    // Main simulation loop
    for (int step = 0; step < steps; step++) {
        // Calculate forces in parallel
        calculate_forces(bodies, total_n, rank, size,
                        local_forces_x, local_forces_y);
        
        // Reduce forces from all processes
        reduce_forces(local_forces_x, local_forces_y,
                     global_forces_x, global_forces_y, total_n);
        
        // Update positions and velocities using Verlet integration
        update_bodies_verlet(bodies, total_n, 
                           global_forces_x, global_forces_y, dt);
    }
    
    // Synchronize all processes before measuring final time
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double execution_time = end_time - start_time;
    
    if (rank == 0) {
        printf("Execution time for parallel: %.6f seconds\n", execution_time);
        printf("Time per simulation step: %.6f seconds\n", execution_time / steps);
        
        // Save results to output file
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "output_parallel_%s", argv[1]);
        save_results(output_filename, bodies, total_n, rank);
        printf("Results saved to %s\n", output_filename);
    }
    
    // Clean up memory
    free(bodies);
    free(local_forces_x);
    free(local_forces_y);
    free(global_forces_x);
    free(global_forces_y);
    
    MPI_Finalize();
    return 0;
}