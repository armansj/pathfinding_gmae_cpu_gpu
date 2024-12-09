import pygame
import sys
import time
import torch
from heapq import heappop, heappush

pygame.init()

WIDTH, HEIGHT = 1200, 800
GRID_SIZE = 60

maze = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1],
    [1, 1, 1, 1, 0, 1, 0, 1, 1, 1,0,0,1,1,1,0,0,0,0,1,1,1,1,1,1,1],
    [1, 0, 0, 0, 0, 1, 0, 1, 0, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1, 1, 1, 1, 0, 1, 1, 1, 0, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1, 1, 1, 1, 1, 1, 0, 1, 0, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1],
    [1, 1, 1, 0, 1, 1, 1, 1, 0, 1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1, 1, 1, 1, 0, 1, 0, 1, 0, 1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1, 1, 1, 1, 0, 1, 1, 1, 0, 1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
]

player_pos = [1, 1]
goal_pos = [9, 12]

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Maze Game")

def draw_maze():
    for y, row in enumerate(maze):
        for x, cell in enumerate(row):
            color = (0, 0, 0) if cell == 1 else (173, 216, 230)
            pygame.draw.rect(screen, color, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

def draw_player():
    pygame.draw.rect(screen, (0, 0, 255), (player_pos[0] * GRID_SIZE, player_pos[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))

def check_win():
    return player_pos == goal_pos

def astar_pathfinding(maze, start, goal, device):
    rows, cols = len(maze), len(maze[0])
    start = torch.tensor(start, device=device)
    goal = torch.tensor(goal, device=device)

    def heuristic(pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    open_set = []
    heappush(open_set, (0, tuple(start.tolist())))

    g_score = torch.full((rows, cols), float('inf'), device=device)
    g_score[start[1], start[0]] = 0

    came_from = {}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while open_set:
        _, current = heappop(open_set)
        current = torch.tensor(current, device=device)

        if torch.equal(current, goal):
            path = []
            while tuple(current.tolist()) in came_from:
                path.append(tuple(current.tolist()))
                current = torch.tensor(came_from[tuple(current.tolist())], device=device)
            path.reverse()
            return path

        for dx, dy in directions:
            neighbor = current + torch.tensor([dx, dy], device=device)
            x, y = neighbor[0].item(), neighbor[1].item()

            if 0 <= x < cols and 0 <= y < rows and maze[y][x] == 0:
                tentative_g_score = g_score[current[1], current[0]] + 1
                if tentative_g_score < g_score[neighbor[1], neighbor[0]]:
                    g_score[neighbor[1], neighbor[0]] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, goal)
                    heappush(open_set, (f_score, tuple(neighbor.tolist())))
                    came_from[tuple(neighbor.tolist())] = tuple(current.tolist())

    return None

start_cpu = time.time()
cpu_path = astar_pathfinding(maze, player_pos, goal_pos, torch.device("cpu"))
end_cpu = time.time()

start_gpu = time.time()
gpu_path = astar_pathfinding(maze, player_pos, goal_pos, torch.device("mps"))
end_gpu = time.time()

print(f"CPU Time: {end_cpu - start_cpu:.6f} seconds")
print(f"GPU Time: {end_gpu - start_gpu:.6f} seconds")

running = True
path_index = 0
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    if cpu_path and path_index < len(cpu_path):
        player_pos = list(cpu_path[path_index])
        path_index += 1
        time.sleep(0.2)

    screen.fill((255, 255, 255))
    draw_maze()
    draw_player()

    if check_win():
        font = pygame.font.Font(None, 36)
        text = font.render("You Win!", True, (0, 255, 0))
        screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - text.get_height() // 2))

    pygame.display.flip()
    pygame.time.Clock().tick(30)
