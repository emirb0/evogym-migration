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

os.environ['OPENAI_API_KEY'] = "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"
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

            # Elbow method logging (optional)
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

            # KMeans with retry to avoid empty clusters
            for attempt in range(10):
                kmeans = KMeans(n_clusters=num_islands, n_init=10, random_state=attempt)
                labels = kmeans.fit_predict(X)
                label_counts = Counter(labels)
                if len(label_counts) == num_islands:
                    break
            else:
                raise ValueError("KMeans failed to produce non-empty clusters after 10 attempts")

            # Log cluster assignments
            with open("logs/kmeans_cluster_log.txt", "w") as f:
                for idx, label in enumerate(labels):
                    f.write(f"Robot {idx}: Cluster {label}\n")
            print("[LOGGED] KMeans cluster assignments.")

            # PCA plot
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

            # Cluster-wise averages
            num_voxel_types = 5
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
            avg_positions = np.divide(
                cluster_voxel_positions,
                cluster_voxel_counts[:, :, None],
                where=cluster_voxel_counts[:, :, None] != 0
            )

            # Plot average voxel counts
            plt.figure(figsize=(10, 6))
            bar_width = 0.15
            x = np.arange(num_voxel_types)
            for i in range(num_islands):
                plt.bar(x + i * bar_width, avg_counts[i], width=bar_width, label=f'Cluster {i}')
            plt.xticks(
                x + bar_width * (num_islands - 1) / 2,
                ['Empty (0)', 'Rigid (1)', 'Soft (2)', 'Act-H (3)', 'Act-V (4)']
            )
            plt.ylabel("Avg Voxel Count per Robot")
            plt.title("Cluster-wise Avg Voxel Type Counts")
            plt.legend()
            plt.tight_layout()
            plt.savefig("logs/cluster_voxel_counts.png")
            print("[SAVED] Voxel counts per cluster: logs/cluster_voxel_counts.png")
            plt.close()

            # Plot average positions
            for voxel_type in range(num_voxel_types):
                plt.figure(figsize=(6, 6))
                for i in range(num_islands):
                    pos = avg_positions[i, voxel_type]
                    plt.scatter(pos[1], 4 - pos[0], label=f"Cluster {i}", s=200)
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

            islands = [[] for _ in range(num_islands)]
            for label, robot in zip(labels, structures):
                islands[label].append(robot)

            return islands

        # ===========================================================
        #          LLM-guided migration WITH fitness info, no fixed objective
        # ===========================================================
        def migrate_robots(islands):
            """
            LLM-based migration:
            - Receives structure + fitness for each robot in each island.
            - Not told to maximize diversity or any specific metric.
            - Free to discover its own implicit migration objective.
            """
            print("[LLM] Deciding robot migrations via LLM...")

            # Build summary with structure + fitness
            summary = {}
            for island_id, island in enumerate(islands):
                summary[island_id] = []
                for robot in island:
                    # ASSUMPTION: each robot has .body and .fitness attributes after PPO training.
                    summary[island_id].append({
                        "body": robot.body.tolist(),
                        "fitness": float(robot.fitness)
                    })

            # Build LLM prompt
            user_prompt = (
                "You manage the evolution of soft robots across several islands. "
                "Each island contains robots represented by 5x5 voxel grids, and each robot has a fitness score. \n"
                "Your task is to decide which robots should migrate between islands in order to improve long-term "
                "evolutionary progress. There is NO predefined migration objective. You may consider factors such as:\n"
                "- High or low fitness\n"
                "- Structural redundancy within an island\n"
                "- Complementarity between islands\n"
                "- Any other internal criterion you decide\n\n"
                "Your output must be a JSON dictionary where each key is a source island ID (integer as a string), "
                "and each value is a list of [robot_index_within_source_island, destination_island_id].\n"
                "Example format:\n"
                "{ \"0\": [[1, 2], [3, 1]], \"2\": [[0, 1]] }\n\n"
                "If you do not want to migrate any robots from an island, you may omit that island from the JSON, "
                "or give it an empty list.\n\n"
                "Here is the current population:\n"
            )

            for island_id, robots in summary.items():
                user_prompt += f"\nIsland {island_id}:\n"
                for idx, r in enumerate(robots):
                    user_prompt += f"Robot {idx}:\n"
                    user_prompt += f"  Structure: {r['body']}\n"
                    user_prompt += f"  Fitness: {r['fitness']}\n"

            user_prompt += "\nReturn ONLY the JSON dictionary, with no explanation.\n"

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You decide robot migrations for a multi-island evolutionary system."
                        },
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.7
                )

                llm_reply = response.choices[0].message.content.strip()
                print("LLM Migration Decision:\n", llm_reply)

                import json
                migration_plan = json.loads(llm_reply)

                # Build new islands
                new_islands = [[] for _ in range(len(islands))]
                migrating = {i: set() for i in range(len(islands))}

                for from_island, moves in migration_plan.items():
                    from_island_int = int(from_island)
                    if from_island_int < 0 or from_island_int >= len(islands):
                        continue
                    for move in moves:
                        if not isinstance(move, list) or len(move) != 2:
                            continue
                        idx, to_island = move
                        idx = int(idx)
                        to_island = int(to_island)
                        if (
                            0 <= from_island_int < len(islands)
                            and 0 <= to_island < len(islands)
                            and 0 <= idx < len(islands[from_island_int])
                        ):
                            migrating[from_island_int].add(idx)
                            new_islands[to_island].append(islands[from_island_int][idx])

                # Add all non-migrating robots back to their original islands
                for island_id, island in enumerate(islands):
                    for idx, robot in enumerate(island):
                        if idx not in migrating[island_id]:
                            new_islands[island_id].append(robot)

                return new_islands

            except Exception as e:
                print(f"[ERROR] LLM migration failed. Falling back to random migration. Error: {e}")
                return migrate_robots_fallback(islands)

        # ===========================================================
        # Fallback migration (keeps a rough 20% migration rate)
        # ===========================================================
        def migrate_robots_fallback(islands, migration_rate=0.2):
            print("[FALLBACK] Random migration")
            new_islands = [[] for _ in range(len(islands))]
            if len(islands) == 0:
                return islands

            for i, island in enumerate(islands):
                if len(island) == 0:
                    continue
                total_to_migrate = max(1, int(len(island) * migration_rate))
                migrants = random.sample(island, min(total_to_migrate, len(island)))
                for robot in migrants:
                    dest = random.choice([j for j in range(len(islands)) if j != i])
                    new_islands[dest].append(robot)
                remaining = [r for r in island if r not in migrants]
                new_islands[i].extend(remaining)

            return new_islands

        # ===========================================================
        # Main evolutionary setup
        # ===========================================================
        num_islands = 4
        max_generations = 50
        total_population = 90
        generations_done = [0] * num_islands

        # Create initial population and assign to islands
        population_structure_hashes = {}
        structures = []
        while len(structures) < total_population:
            robot, _ = sample_robot((5, 5))
            if (
                is_connected(robot)
                and has_actuator(robot)
                and hashable(robot) not in population_structure_hashes
            ):
                structures.append(robot)
                population_structure_hashes[hashable(robot)] = True

        islands = assign_islands_by_voxel_value(structures, num_islands=num_islands)

        # Logging helper
        def log_voxel_medians(island_id, generation_id, population):
            os.makedirs("logs", exist_ok=True)
            log_file = f"logs/island{island_id}_gen{generation_id}.txt"
            with open(log_file, "w") as f:
                for i, robot in enumerate(population):
                    flat = robot.body.flatten()
                    median = np.median(flat)
                    f.write(f"Robot {i}: median voxel value = {median}\n")
            print(f"[LOGGED] voxel medians for Island {island_id} Gen {generation_id}")

        # ===========================================================
        # Synchronized generation loop with periodic migration
        # ===========================================================
        while max(generations_done) < max_generations:
            for island_id in range(num_islands):
                if generations_done[island_id] >= max_generations:
                    continue

                print(f"▶ Running Island {island_id} - Generation {generations_done[island_id]}")
                evolved_population = run(
                    pop_size=len(islands[island_id]),
                    structure_shape=(5, 5),
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

            # Trigger migration when all islands have completed the same generation
            if all(g % 2 == 0 for g in generations_done) and len(set(generations_done)) == 1:
                print(f">> Migration triggered at generation {generations_done[0]}")
                islands = migrate_robots(islands)

            elif all(g % 2 == 1 for g in generations_done) and len(set(generations_done)) == 1:
                print(f">> Migration triggered at generation {generations_done[0]}")
                islands = migrate_robots(islands)
