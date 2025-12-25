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


        def levenshtein_ratio(seq1, seq2):
            """
            Calculates the Levenshtein distance between two sequences and returns value between 0 and 1.
            (1 means identical, 0 means completely different).
            We interpret 'diversity' as 1 - ratio (so higher is more diverse).
            """
            size_x = len(seq1) + 1
            size_y = len(seq2) + 1
            matrix = np.zeros ((size_x, size_y))
            for x in range(size_x):
                matrix [x, 0] = x
            for y in range(size_y):
                matrix [0, y] = y

            for x in range(1, size_x):
                for y in range(1, size_y):
                    if seq1[x-1] == seq2[y-1]:
                        matrix [x,y] = min(
                            matrix[x-1, y] + 1,
                            matrix[x-1, y-1],
                            matrix[x, y-1] + 1
                        )
                    else:
                        matrix [x,y] = min(
                            matrix[x-1,y] + 1,
                            matrix[x-1,y-1] + 1,
                            matrix[x,y-1] + 1
                        )
            # Distance
            dist = matrix[size_x - 1, size_y - 1]
            max_len = max(len(seq1), len(seq2))
            if max_len == 0: return 1.0 # Identical empty
            return dist

        def calculate_diversity(islands):
            """
            Computes global diversity score based on average pairwise Levenshtein distance.
            Higher score = More diverse.
            """
            island_diversities = []
            for island in islands:
                if len(island) < 2:
                    island_diversities.append(0.0)
                    continue
                
                distances = []
                # Flatten bodies for sequence comparison
                flat_bodies = [r.body.flatten() for r in island]
                
                for i in range(len(flat_bodies)):
                    for j in range(i + 1, len(flat_bodies)):
                        # Levenshtein distance tells us how many edits apart they are
                        # We want to maximize this distance
                        dist = levenshtein_ratio(flat_bodies[i], flat_bodies[j])
                        distances.append(dist)
                
                if distances:
                    island_diversities.append(np.mean(distances))
                else:
                    island_diversities.append(0.0)
            
            # Use mean of island diversities as the global metric
            return np.mean(island_diversities) if island_diversities else 0.0

        def migrate_robots(islands, migration_rate=0.2):
            from LLM.run_llm import client
            
            # 1. Baseline Diversity
            current_diversity = calculate_diversity(islands)
            print(f"[MIGRATION CHECK] Current Diversity (Levenshtein): {current_diversity:.4f}")

            max_retries = 2
            for attempt in range(max_retries + 1):
                print(f"[LLM] Migration Attempt {attempt + 1}...")

                # --- Original Prompt Structure ---
                summary = {}
                for i, island in enumerate(islands):
                    summary[i] = []
                    for robot in island:
                        summary[i].append(robot.body.tolist())

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

                if attempt > 0:
                    user_prompt += f"\n\nWARNING: Your previous proposal was REJECTED because it did not increase structural diversity (Current Score: {current_diversity:.4f}). You must propose different migrations that actually separate similar robots."

                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a migration decision expert for evolving soft robots."},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_tokens=1000,
                        temperature=0.8 + (attempt * 0.1) # Increase temp on retry
                    )
                    llm_reply = response.choices[0].message.content.strip()
                    if llm_reply.startswith("```json"): llm_reply = llm_reply[7:-3]
                    if llm_reply.endswith("```"): llm_reply = llm_reply[:-3]
                    
                    import json
                    migration_plan = json.loads(llm_reply)
                    
                    # --- Simulate Migration ---
                    cand_islands = [list(isl) for isl in islands]
                    migrating_indices = {i: set() for i in range(len(islands))}
                    
                    # 1. Collect migrants
                    migrants = [] 
                    for src_str, moves in migration_plan.items():
                        src = int(src_str)
                        for move in moves:
                            if len(move) != 2: continue
                            idx, dest = int(move[0]), int(move[1])
                            if 0 <= src < len(islands) and 0 <= dest < len(islands) and 0 <= idx < len(islands[src]):
                                migrants.append((src, idx, dest))
                                migrating_indices[src].add(idx)

                    # 2. Rebuild islands
                    new_islands_temp = []
                    for i in range(len(islands)):
                        remaining = [robot for j, robot in enumerate(islands[i]) if j not in migrating_indices[i]]
                        new_islands_temp.append(remaining)
                    
                    # Add to dest
                    for src, idx, dest in migrants:
                        new_islands_temp[dest].append(islands[src][idx])

                    # 3. Evaluate Diversity
                    new_div = calculate_diversity(new_islands_temp)
                    print(f"  > Proposed Diversity: {new_div:.4f} (Delta: {new_div - current_diversity:.4f})")

                    if new_div > current_diversity:
                        print("  [ACCEPTED] Diversity improved.")
                        return new_islands_temp
                    else:
                        print("  [REJECTED] Diversity did not improve.")

                except Exception as e:
                    print(f"[ERROR] Attempt {attempt+1} failed: {e}")
            
            print("[ABORT] All attempts failed. STRICT REJECTION: Maintaining original population.")
            return islands # REJECT MIGRATION

        def migrate_robots_fallback(islands, migration_rate=0.2):
             # Since we want explicit rejection if LLM fails, this fallback is effectively unused
             # by the new logic, but kept for interface compatibility if needed.
             return islands


        num_islands = 4
        max_generations = 50
        total_population = 30
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
