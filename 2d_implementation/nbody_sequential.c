#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define G 6.67430e-11  // Gravitational constant
#define EPSILON 1e-9   // Softening parameter to avoid divisions by zero

typedef struct {
    int id;
    double mass;
    double pos_x, pos_y;
    double vel_x, vel_y;
    double force_x, force_y;
} Body;

// Function to read data from CSV file (2D version)
int read_csv(const char* filename, Body** bodies) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error in opening the file %s\n", filename);
        return -1;
    }
    
    char line[1024];
    int n = 0;
    
    // Count the number of rows (excluding header)
    fgets(line, sizeof(line), file); // Skip header
    while (fgets(line, sizeof(line), file)) {
        n++;
    }
    
    // Allocates memory for bodies
    *bodies = (Body*)malloc(n * sizeof(Body));
    if (!*bodies) {
        printf("Memory allocation error\n");
        fclose(file);
        return -1;
    }
    
    // Go back to the beginning of the file and read the data
    rewind(file);
    fgets(line, sizeof(line), file); // Skip header
    
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
        }
    }
    
    fclose(file);
    return n;
}

// Function to calculate gravitational forces (sequential brute force algorithm, 2D)
void calculate_forces(Body* bodies, int n) {
    // Force reset
    for (int i = 0; i < n; i++) {
        bodies[i].force_x = 0.0;
        bodies[i].force_y = 0.0;
    }
    
    // Calculation of O(n²) forces
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            // Calculate the distance between bodies i and j
            double dx = bodies[j].pos_x - bodies[i].pos_x;
            double dy = bodies[j].pos_y - bodies[i].pos_y;
            
            double dist_squared = dx*dx + dy*dy + EPSILON*EPSILON;
            double dist = sqrt(dist_squared);
            double dist_cubed = dist_squared * dist;
            
            // Calculate the gravitational force
            double force_magnitude = G * bodies[i].mass * bodies[j].mass / dist_cubed;
            
            // Components of the force
            double force_x = force_magnitude * dx;
            double force_y = force_magnitude * dy;
            
            // Applies Newton's third law of motion
            bodies[i].force_x += force_x;
            bodies[i].force_y += force_y;
            
            bodies[j].force_x -= force_x;
            bodies[j].force_y -= force_y;
        }
    }
}

// Function to update positions and velocities (Euler integration, 2D)
void update_bodies(Body* bodies, int n, double dt) {
    for (int i = 0; i < n; i++) {
        // Velocity update
        bodies[i].vel_x += (bodies[i].force_x / bodies[i].mass) * dt;
        bodies[i].vel_y += (bodies[i].force_y / bodies[i].mass) * dt;
        
        // Position update
        bodies[i].pos_x += bodies[i].vel_x * dt;
        bodies[i].pos_y += bodies[i].vel_y * dt;
    }
}

// Function to print the status of bodies (2D version)
void print_bodies(Body* bodies, int n) {
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

// Function to save results to a CSV file (2D version)
void save_results(const char* filename, Body* bodies, int n) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("Error in opening the output file %s\n", filename);
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

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Use: %s <nome_file_csv>\n", argv[0]);
        return 1;
    }
    
    Body* bodies;
    int n = read_csv(argv[1], &bodies);
    
    if (n < 0) {
        return 1;
    }
    
    printf("Read %d bodies from file %s\n\n", n, argv[1]);
    
    // Simulation parameters
    double dt = 0.01;  // Time step
    int steps = 100;   // Number of simulation steps
    
    printf("Initial state:\n");
    print_bodies(bodies, n);
    
    // Measure the execution time
    clock_t start = clock();
    
    // Simulation
    for (int step = 0; step < steps; step++) {
        calculate_forces(bodies, n);
        update_bodies(bodies, n, dt);
        
        // Print every 20 steps
        if (step % 20 == 0) {
            printf("Step %d:\n", step);
            print_bodies(bodies, n);
        }
    }
    
    clock_t end = clock();
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("Execution time for sequential: %.6f seconds\n", cpu_time_used);
    printf("Time to calculate forces per step: %.6f seconds\n", cpu_time_used / steps);
    
    // Save the final results
    char output_filename[256];
    snprintf(output_filename, sizeof(output_filename), "output_%s", argv[1]);
    save_results(output_filename, bodies, n);
    printf("Results saved in %s\n", output_filename);
    
    free(bodies);
    return 0;
}