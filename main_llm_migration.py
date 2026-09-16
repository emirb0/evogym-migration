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

from openai import OpenAI
os.environ['OPENAI_API_KEY'] = "placeholder"
os.environ['OPENAI_BASE_URL'] = "https://api.openai.com/v1/"
client = OpenAI()

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
            print("[LLM] Deciding robot migrations via LLM...")

            # Prepare data for LLM: summarize each robot structure
            summary = {}
            for i, island in enumerate(islands):
                summary[i] = []
                for robot in island:
                    summary[i].append(robot.body.tolist())

            from LLM.run_llm import client  # assumes OpenAI client is setup like in run_llm.py

            user_prompt = (
                "You are an intelligent system managing the evolution of soft robots across several islands. "
                "Each island contains a number of robots, represented by 5x5 matrices. "
                "Each cell in a matrix is a voxel with a value from 0 to 4, indicating the voxel type:\n"
                "0 = Empty, 1 = Rigid, 2 = Soft, 3 = Horizontal Actuator, 4 = Vertical Actuator.\n\n"

                "Your task is to improve **structural diversity within each island**. "
                "To do this, identify robots that are too similar to others in their island, and migrate them to other islands.\n"
                "This helps ensure each island has a more varied set of robot designs.\n\n"

                "You may migrate up to 20% of the robots in each island. "
                "Provide your decision as a JSON dictionary with this format:\n"
                "{\n"
                "  \"0\": [[0, 1], [2, 3]],\n"
                "  \"1\": [[3, 0]]\n"
                "}\n"
                "This means: From island 0, robot 0 goes to island 1, and robot 2 goes to island 3. From island 1, robot 3 goes to island 0.\n\n"

                "Here is the current robot distribution across islands:\n"
            )

            for island_id, robots in summary.items():
                user_prompt += f"\nIsland {island_id}:\n"
                for r in robots:
                    user_prompt += str(r) + "\n"

            user_prompt += (
                "\nOnly return the JSON dictionary. Do not provide any explanation or additional comments."
            )


            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": "You are a migration decision expert for evolving soft robots."},
                            {"role": "user", "content": user_prompt}],
                    max_tokens=1000,
                    temperature=0.8
                )
                llm_reply = response.choices[0].message.content.strip()
                print("LLM Migration Decision:\n", llm_reply)

                # Safe evaluation of JSON-like structure
                import json
                migration_plan = json.loads(llm_reply)

                new_islands = [[] for _ in range(len(islands))]

                # Track robots that are migrating to avoid duplication
                migrating = {i: set() for i in range(len(islands))}

                for from_island, migrations in migration_plan.items():
                    from_island = int(from_island)
                    for move in migrations:
                        if not isinstance(move, list) or len(move) != 2:
                            raise ValueError(f"Invalid migration format: {move}")
                        idx, to_island = move
                        idx = int(idx)
                        to_island = int(to_island)
                        if 0 <= idx < len(islands[from_island]) and 0 <= to_island < len(islands):
                            migrating[from_island].add(idx)
                            new_islands[to_island].append(islands[from_island][idx])


                for i, island in enumerate(islands):
                    for j, robot in enumerate(island):
                        if j not in migrating[i]:
                            new_islands[i].append(robot)

                return new_islands

            except Exception as e:
                print(f"[ERROR] LLM migration failed. Falling back to random migration. Error: {e}")
                return migrate_robots_fallback(islands, migration_rate)


        def migrate_robots_fallback(islands, migration_rate=0.2):
            print("[FALLBACK] Random migration")
            new_islands = [[] for _ in range(len(islands))]
            total_to_migrate = int(len(islands[0]) * migration_rate)
            for i, island in enumerate(islands):
                migrants = random.sample(island, min(total_to_migrate, len(island)))
                for robot in migrants:
                    dest = random.choice([j for j in range(len(islands)) if j != i])
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

            elif all(g % 2 == 1 for g in generations_done) and len(set(generations_done)) == 1:
                print(f">> Migration triggered at generation {generations_done[0]}")
                islands = migrate_robots(islands, migration_rate=0.2)
