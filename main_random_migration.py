import random
import time
import numpy as np
from LLM.run_llm import run
from ppo.arguments import get_args
import torch
import os
from collections import defaultdict, Counter
from evogym import sample_robot, hashable
from evogym.utils import is_connected, has_actuator, get_full_connectivity
from utils.algo_utils import Structure
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    args = get_args()
    args.num_processes = 4
    args.cuda = True
    args.no_cuda = False
    args.eval_interval = 50
    tasks = ["Carrier-v0"]

    for task in tasks:

        def assign_islands_by_voxel_value(structures, num_islands=4):
            os.makedirs("logs", exist_ok=True)
            num_voxel_types = 5
            robot_features = []

            for robot in structures:
                features = []
                for voxel_type in range(num_voxel_types):
                    count = 0
                    pos_sum = np.array([0.0, 0.0])
                    for i in range(5):
                        for j in range(5):
                            if int(robot[i, j]) == voxel_type:
                                count += 1
                                pos_sum += np.array([i, j])
                    avg_pos = pos_sum / count if count > 0 else np.array([0.0, 0.0])
                    features.append(count)
                    features.extend(avg_pos.tolist())
                robot_features.append(features)

            X = np.array(robot_features)

            # === ELBOW METHOD TO DECIDE CLUSTER COUNT ===
            #if num_islands == 'Dynamic':
            sse = []
            K_range = range(1, 11)
            for k in K_range:
                kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
                kmeans.fit(X)
                sse.append(kmeans.inertia_)
            plt.figure(figsize=(8, 5))
            plt.plot(K_range, sse, 'o-', markerfacecolor='red')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Sum of Squared Errors (SSE)')
            plt.title('Elbow Method: SSE vs. Number of Clusters')
            plt.grid(True)
            plt.savefig("logs/elbow_sse_plot.png")
            print("[SAVED] Elbow plot saved to logs/elbow_sse_plot.png")
            plt.close()
            # Choose default (elbow point visually or fixed for now)


            # KMeans with retry
            for attempt in range(10):
                kmeans = KMeans(n_clusters=num_islands, n_init=10, random_state=attempt)
                labels = kmeans.fit_predict(X)
                label_counts = Counter(labels)
                if len(label_counts) == num_islands:
                    break
            else:
                raise ValueError("KMeans failed to produce non-empty clusters after 10 attempts")

            # === LOGGING CLUSTER ASSIGNMENTS ===
            with open("logs/kmeans_cluster_log.txt", "w") as f:
                for idx, label in enumerate(labels):
                    f.write(f"Robot {idx}: Cluster {label}\n")
            print("[LOGGED] KMeans cluster assignments.")

            # === PCA PLOT ===
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
            plt.figure(figsize=(8, 6))
            plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', edgecolors='k')
            plt.title("PCA - Robot Clustering")
            plt.xlabel("PCA Component 1")
            plt.ylabel("PCA Component 2")
            plt.savefig("logs/kmeans_pca.png")
            print("[SAVED] PCA plot at logs/kmeans_pca.png")
            plt.close()

            # === CLUSTER-WISE AVERAGES ===
            cluster_voxel_counts = np.zeros((num_islands, num_voxel_types))
            cluster_voxel_positions = np.zeros((num_islands, num_voxel_types, 2))
            cluster_sizes = [0] * num_islands

            for robot, label in zip(structures, labels):
                cluster_sizes[label] += 1
                for voxel_type in range(num_voxel_types):
                    count = 0
                    pos_sum = np.array([0.0, 0.0])
                    for i in range(5):
                        for j in range(5):
                            if int(robot[i, j]) == voxel_type:
                                count += 1
                                pos_sum += np.array([i, j])
                    cluster_voxel_counts[label, voxel_type] += count
                    if count > 0:
                        cluster_voxel_positions[label, voxel_type] += pos_sum

            avg_counts = cluster_voxel_counts / np.array(cluster_sizes)[:, None]
            avg_positions = np.divide(cluster_voxel_positions, cluster_voxel_counts[:, :, None], where=cluster_voxel_counts[:, :, None] != 0)

            # === PLOT AVERAGE VOXEL COUNTS ===
            plt.figure(figsize=(10, 6))
            bar_width = 0.15
            x = np.arange(num_voxel_types)
            for i in range(num_islands):
                plt.bar(x + i * bar_width, avg_counts[i], width=bar_width, label=f'Cluster {i}')
            plt.xticks(x + bar_width * (num_islands - 1) / 2, ['Empty (0)', 'Rigid (1)', 'Soft (2)', 'Act-H (3)', 'Act-V (4)'])
            plt.ylabel("Avg Voxel Count per Robot")
            plt.title("Cluster-wise Avg Voxel Type Counts")
            plt.legend()
            plt.tight_layout()
            plt.savefig("logs/cluster_voxel_counts.png")
            print("[SAVED] Voxel counts per cluster: logs/cluster_voxel_counts.png")
            plt.close()

            # === PLOT AVERAGE POSITIONS ===
            for voxel_type in range(num_voxel_types):
                plt.figure(figsize=(6, 6))
                for i in range(num_islands):
                    pos = avg_positions[i, voxel_type]
                    plt.scatter(pos[1], 4 - pos[0], label=f"Cluster {i}", s=200)  # y-axis inverted
                plt.xlim(-0.5, 4.5)
                plt.ylim(-0.5, 4.5)
                plt.grid(True)
                plt.title(f"Average Position of Voxel Type {voxel_type}")
                plt.xlabel("Column")
                plt.ylabel("Row")
                plt.gca().set_aspect('equal')
                plt.legend()
                plt.savefig(f"logs/voxel_type_{voxel_type}_avg_positions.png")
                print(f"[SAVED] Avg position of voxel {voxel_type} at logs/voxel_type_{voxel_type}_avg_positions.png")
                plt.close()

            # === Final Assignment ===
            islands = [[] for _ in range(num_islands)]
            for label, robot in zip(labels, structures):
                islands[label].append(robot)

            return islands


        def migrate_robots(islands, migration_rate=0.2):
            new_islands = [[] for _ in range(len(islands))]
            total_to_migrate = int(len(islands[0]) * migration_rate)
            for i, island in enumerate(islands):
                migrants = random.sample(island, min(total_to_migrate, len(island)))
                for robot in migrants:
                    dest = random.choice([j for j in range(len(islands)) if j != i])
                    print(f"Robot from Island {i} migrating to Island {dest}")
                    new_islands[dest].append(robot)
                remaining = [r for r in island if r not in migrants]
                new_islands[i].extend(remaining)
            return new_islands

        num_islands = 4
        max_generations = 50
        total_population = 90
        generations_done = [0] * num_islands

        # === Create initial population and assign to islands ===
        population_structure_hashes = {}
        structures = []
        while len(structures) < total_population:
            robot, _ = sample_robot((5, 5))
            if is_connected(robot) and has_actuator(robot) and hashable(robot) not in population_structure_hashes:
                structures.append(robot)
                population_structure_hashes[hashable(robot)] = True

        islands = assign_islands_by_voxel_value(structures, num_islands=num_islands)

        
        def log_voxel_medians(island_id, generation_id, population):
            os.makedirs("logs", exist_ok=True)
            log_file = f"logs/island{island_id}_gen{generation_id}.txt"
            with open(log_file, "w") as f:
                for i, robot in enumerate(population):
                    flat = robot.body.flatten()
                    median = np.median(flat)
                    f.write(f"Robot {i}: median voxel value = {median}\n")
            print(f"[LOGGED] voxel medians for Island {island_id} Gen {generation_id}")
    

        # === Synchronized generation loop ===
        while max(generations_done) < max_generations:
            for island_id in range(num_islands):
                if generations_done[island_id] >= max_generations:
                    continue
                print(f"▶ Running Island {island_id} - Generation {generations_done[island_id]}")
                evolved_population = run(
                    pop_size=len(islands[island_id]),
                    structure_shape=(5,5),
                    experiment_name=task + f"-Island{island_id}-Gen{generations_done[island_id]}",
                    max_evaluations=1000,
                    train_iters=1000,
                    num_cores=4,
                    env_name=task,
                    args=args,
                    population=islands[island_id],
                    num_generations=1
                )
                islands[island_id] = evolved_population
                log_voxel_medians(island_id, generations_done[island_id], evolved_population)
                generations_done[island_id] += 1

            if all(g % 2 == 0 for g in generations_done) and len(set(generations_done)) == 1:
                print(f">> Migration triggered at generation {generations_done[0]}")
                islands = migrate_robots(islands, migration_rate=0.2)
