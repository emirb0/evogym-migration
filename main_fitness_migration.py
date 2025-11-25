import random
import time
import numpy as np
from LLM.run_llm import run
from ppo.arguments import get_args
import torch
import os
from collections import defaultdict
from evogym import sample_robot, hashable
from evogym.utils import is_connected, has_actuator, get_full_connectivity
from utils.algo_utils import Structure

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

        def assign_islands_by_voxel_value(structures, num_islands=3):
            islands = [[] for _ in range(num_islands)]
            for robot in structures:
                voxels = robot.flatten()
                avg_voxel_value = np.mean(voxels)
                if avg_voxel_value <= 1.5:
                    islands[0].append(robot)
                elif avg_voxel_value <= 2.0:
                    islands[1].append(robot)
                else:
                    islands[2].append(robot)
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

        num_islands = 3
        max_generations = 4
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
                    max_evaluations=50,
                    train_iters=100,
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
