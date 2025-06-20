import random
import csv
import os

def generate_bodies(n_bodies, bounds=(-100, 100), mass_range=(1, 100)):
    """
    Generates n_bodies bodies with random positions, velocities, and masses.
    """
    bodies = []
    for i in range(n_bodies):
        body = {
            'id': i,
            'mass': random.uniform(mass_range[0], mass_range[1]),
            'position': {
                'x': random.uniform(bounds[0], bounds[1]),
                'y': random.uniform(bounds[0], bounds[1]),
                'z': random.uniform(bounds[0], bounds[1])
            },
            'velocity': {
                'x': random.uniform(bounds[0] / 10, bounds[1] / 10),
                'y': random.uniform(bounds[0] / 10, bounds[1] / 10),
                'z': random.uniform(bounds[0] / 10, bounds[1] / 10)
            }
        }
        bodies.append(body)
    return bodies

def save_bodies_to_file(bodies, filename):
    """
    Save the bodies to a CSV file in the dataset folder.
    """
    os.makedirs("dataset", exist_ok=True)
    filepath = os.path.join("dataset", filename)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'mass', 'pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z'])
        for body in bodies:
            writer.writerow([
                body['id'], body['mass'],
                body['position']['x'], body['position']['y'], body['position']['z'],
                body['velocity']['x'], body['velocity']['y'], body['velocity']['z']
            ])

def load_bodies_from_file(filename):
    """
    Load bodies from a CSV file in the dataset folder.
    """
    filepath = os.path.join("dataset", filename)
    bodies = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            body = {
                'id': int(row['id']),
                'mass': float(row['mass']),
                'position': {
                    'x': float(row['pos_x']),
                    'y': float(row['pos_y']),
                    'z': float(row['pos_z'])
                },
                'velocity': {
                    'x': float(row['vel_x']),
                    'y': float(row['vel_y']),
                    'z': float(row['vel_z'])
                }
            }
            bodies.append(body)
    return bodies

# Example of use
if __name__ == "__main__":
    test_sizes = [10, 100, 500, 1000, 2000, 3000, 5000, 10000]
    for n in test_sizes:
        print(f"Generando {n} corpi...")
        bodies = generate_bodies(n)
        filename = f'bodies_{n}.csv'
        save_bodies_to_file(bodies, filename)
        print(f"Saved dataset with {n} bodies in dataset/{filename}")

    print("\nLoading test...")
    bodies_test = load_bodies_from_file('bodies_100.csv')
    print(f"Loaded {len(bodies_test)} bodies from CSV file")
    print(f"First body: {bodies_test[0]}")
