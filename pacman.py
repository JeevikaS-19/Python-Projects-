import pygame
import sys
import random

# Initialize Pygame
pygame.init()

# Constants
TILE_SIZE = 24
GRID_WIDTH = 40
GRID_HEIGHT = 31
SCREEN_WIDTH = GRID_WIDTH * TILE_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * TILE_SIZE + 50
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
PINK = (255, 182, 193)
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)
DARK_BLUE = (0, 0, 139) # Frightened color

# Map Layout (1 = Wall, 0 = Pellet, 2 = Empty, 9 = Ghost House Gate, 8 = Ghost House Interior, 3 = Power Pellet)
# Consistent 40x31 Map
GAME_MAP = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 3, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 3, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 9, 9, 1, 1, 1, 1, 1, 1, 9, 9, 1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1, 1, 1, 1, 1, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 3, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 3, 1, 0, 1],
    [1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
    [1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

class Player(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
        self.rect = self.image.get_rect()
        self.rect.x = x * TILE_SIZE
        self.rect.y = y * TILE_SIZE
        self.speed = 2 # Reduced speed from 3
        self.direction = (0, 0)
        self.next_direction = (0, 0)
        self.start_pos = (self.rect.x, self.rect.y)
        self.score = 0
        self.lives = 3
        # Animation
        self.mouth_open = 45
        self.mouth_speed = 5
        self.mouth_closing = True
        self.rotation = 0

    def update(self, walls):
        # Animation
        if self.mouth_closing:
            self.mouth_open -= self.mouth_speed
            if self.mouth_open <= 0:
                self.mouth_closing = False
                self.mouth_open = 0
        else:
            self.mouth_open += self.mouth_speed
            if self.mouth_open >= 45:
                self.mouth_closing = True
                self.mouth_open = 45

        # Rotation based on direction
        if self.direction == (1, 0): self.rotation = 0
        elif self.direction == (-1, 0): self.rotation = 180
        elif self.direction == (0, -1): self.rotation = 90
        elif self.direction == (0, 1): self.rotation = 270

        # Draw Pacman
        self.image.fill((0, 0, 0, 0)) # Clear
        radius = TILE_SIZE // 2 - 2
        center = (TILE_SIZE // 2, TILE_SIZE // 2)
        
        # Check alignment and mouth
        # Simple Draw:
        self.image.fill((0,0,0,0))
        pts = [center]
        import math
        start_angle = self.mouth_open
        end_angle = 360 - self.mouth_open
        
        for angle in range(start_angle, end_angle + 1, 10):
            rad = math.radians(angle)
            x = center[0] + radius * math.cos(rad)
            y = center[1] - radius * math.sin(rad)
            pts.append((x, y))
        pts.append(center)
        
        if len(pts) > 2:
            pygame.draw.polygon(self.image, YELLOW, pts)
        
        if self.rotation != 0:
            self.image = pygame.transform.rotate(self.image, self.rotation)
            old_center = self.rect.center
            self.rect = self.image.get_rect()
            self.rect.center = old_center

        # MOVEMENT
        if self.next_direction != (0, 0):
            old_x, old_y = self.rect.x, self.rect.y
            self.rect.x += self.next_direction[0] * self.speed
            self.rect.y += self.next_direction[1] * self.speed
            if not pygame.sprite.spritecollide(self, walls, False):
                self.direction = self.next_direction
                self.next_direction = (0, 0)
            else:
                self.rect.x, self.rect.y = old_x, old_y

        self.rect.x += self.direction[0] * self.speed
        if pygame.sprite.spritecollide(self, walls, False):
            self.rect.x -= self.direction[0] * self.speed
        
        self.rect.y += self.direction[1] * self.speed
        if pygame.sprite.spritecollide(self, walls, False):
            self.rect.y -= self.direction[1] * self.speed

        # Screen wrap
        if self.rect.right < 0:
            self.rect.left = SCREEN_WIDTH
        elif self.rect.left > SCREEN_WIDTH:
            self.rect.right = 0

    def reset(self):
        self.rect.x, self.rect.y = self.start_pos
        self.direction = (0, 0)
        self.next_direction = (0, 0)
        self.rotation = 0

class Ghost(pygame.sprite.Sprite):
    def __init__(self, x, y, color):
        super().__init__()
        self.base_color = color
        self.image = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
        self.rect = self.image.get_rect()
        self.rect.x = x * TILE_SIZE
        self.rect.y = y * TILE_SIZE
        self.speed = 2 # Reduced speed
        self.direction = (random.choice([1, -1]), 0) # Start moving!
        self.start_pos = (self.rect.x, self.rect.y)
        self.frightened = False
        self.frightened_timer = 0
        self.draw_ghost()

    def draw_ghost(self):
        self.image.fill((0,0,0,0))
        
        color = self.base_color
        if self.frightened:
            # Flicker if timer is low (last 2 seconds = 120 frames)
            if self.frightened_timer < 180 and (self.frightened_timer // 10) % 2 == 0:
                 color = WHITE # Flash white
            else:
                 color = DARK_BLUE # Frightened blue
        
        radius = TILE_SIZE // 2 - 2
        center_x = TILE_SIZE // 2
        center_y = TILE_SIZE // 2
        
        pygame.draw.circle(self.image, color, (center_x, center_y), radius)
        rect_body = pygame.Rect(center_x - radius, center_y, radius * 2, radius)
        pygame.draw.rect(self.image, color, rect_body)
        
        feet_radius = radius // 3
        for i in range(3):
            f_x = (center_x - radius) + i * (feet_radius * 2) + feet_radius
            f_y = center_y + radius
            pygame.draw.circle(self.image, color, (int(f_x), int(f_y)), int(feet_radius))

        eye_radius = 4
        pupil_radius = 2
        eye_offset_x = 4
        eye_offset_y = -2
        p_dx, p_dy = self.direction[0] * 2, self.direction[1] * 2

        # In frightened mode, maybe simple face or flashing
        if self.frightened:
            # Simple faint white eyes or zigzag mouth
            # If flashing white, normal eyes might look better or just stick to "scared" face
            if color == WHITE: # Flashing back to normal-ish look but still scared face maybe?
                 # Draw normal eyes but with red pupils maybe?
                 pygame.draw.circle(self.image, BLACK, (center_x - eye_offset_x, center_y + eye_offset_y), 2)
                 pygame.draw.circle(self.image, BLACK, (center_x + eye_offset_x, center_y + eye_offset_y), 2)
                 pygame.draw.line(self.image, BLACK, (center_x - 6, center_y + 6), (center_x + 6, center_y + 6), 2)
            else:
                 # Scared face
                 pygame.draw.circle(self.image, WHITE, (center_x - eye_offset_x, center_y + eye_offset_y), 2)
                 pygame.draw.circle(self.image, WHITE, (center_x + eye_offset_x, center_y + eye_offset_y), 2)
                 pygame.draw.line(self.image, WHITE, (center_x - 6, center_y + 6), (center_x + 6, center_y + 6), 2)
        else:
            pygame.draw.circle(self.image, WHITE, (center_x - eye_offset_x, center_y + eye_offset_y), eye_radius)
            pygame.draw.circle(self.image, WHITE, (center_x + eye_offset_x, center_y + eye_offset_y), eye_radius)
            
            pygame.draw.circle(self.image, BLUE, (center_x - eye_offset_x + p_dx, center_y + eye_offset_y + p_dy), pupil_radius)
            pygame.draw.circle(self.image, BLUE, (center_x + eye_offset_x + p_dx, center_y + eye_offset_y + p_dy), pupil_radius)

    def update(self, walls):
        # manage timer
        if self.frightened:
            self.frightened_timer -= 1
            if self.frightened_timer <= 0:
                self.frightened = False
                # Re-align to grid if speed changes (1 vs 2) but we are staying 2 for now, just slower logic maybe?
                # Actually, standard behavior is half speed. 
                # Since simple collision, let's keep speed same but changing direction more randomly
        
        # Move logic
        current_speed = 1 if self.frightened else self.speed # Slower when frightened (1 pixel per frame vs 2)

        for _ in range(current_speed):
            self.rect.x += self.direction[0]
            self.rect.y += self.direction[1]
            
            hit_wall = pygame.sprite.spritecollide(self, walls, False)
            
            center_x = self.rect.x + self.rect.width // 2
            center_y = self.rect.y + self.rect.height // 2
            tile_x = center_x // TILE_SIZE
            tile_y = center_y // TILE_SIZE
            is_centered_x = abs(center_x - (tile_x * TILE_SIZE + TILE_SIZE // 2)) <= 1 # tighter tolerance for speed 1
            is_centered_y = abs(center_y - (tile_y * TILE_SIZE + TILE_SIZE // 2)) <= 1
            
            if hit_wall or (is_centered_x and is_centered_y and random.random() < 0.2): # Higher turn rate
                if hit_wall:
                    self.rect.x -= self.direction[0]
                    self.rect.y -= self.direction[1]
                
                possible_directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                random.shuffle(possible_directions)
                for dx, dy in possible_directions:
                    temp_rect = self.rect.copy()
                    temp_rect.x += dx * TILE_SIZE
                    temp_rect.y += dy * TILE_SIZE
                    if not self.check_collision_dummy(temp_rect, walls) and (dx, dy) != (-self.direction[0], -self.direction[1]):
                         self.direction = (dx, dy)
                         break
                else:
                     self.direction = (-self.direction[0], -self.direction[1])
            
            if hit_wall: # Break inner loop if hit wall to avoidgetting stuck in wall
                 break

        self.draw_ghost() # Refresh visuals (frightened state etc)

        if self.rect.right < 0:
            self.rect.left = SCREEN_WIDTH
        elif self.rect.left > SCREEN_WIDTH:
            self.rect.right = 0
            
    def check_collision_dummy(self, rect, walls):
        for wall in walls:
            if rect.colliderect(wall.rect):
                return True
        return False

    def reset(self):
        self.rect.x, self.rect.y = self.start_pos
        self.direction = (random.choice([1, -1]), 0)
        self.frightened = False
        self.frightened_timer = 0
    
    def scare(self):
        self.frightened = True
        self.frightened_timer = 600 # 10 seconds at 60 FPS
        # Reverse direction immediately
        self.direction = (-self.direction[0], -self.direction[1])

class Wall(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((TILE_SIZE, TILE_SIZE))
        self.image.fill(BLUE)
        # Draw some lines to make it look nicer (hollow walls)
        pygame.draw.rect(self.image, BLACK, (4, 4, TILE_SIZE-8, TILE_SIZE-8))
        self.rect = self.image.get_rect()
        self.rect.x = x * TILE_SIZE
        self.rect.y = y * TILE_SIZE

class Pellet(pygame.sprite.Sprite):
    def __init__(self, x, y, is_power=False):
        super().__init__()
        self.is_power = is_power
        size = 12 if is_power else 6
        self.image = pygame.Surface((size, size), pygame.SRCALPHA)
        color = PINK if not is_power else WHITE  # Power pellets are white/bigger? or Red 'Cherry'?
        # User requested Cherries. Let's make it a red circle for now to look like a cherry
        if is_power:
            # Draw simple cherry (two red dots connected)
            pygame.draw.circle(self.image, RED, (size//4, size//1.5), size//3)
            pygame.draw.circle(self.image, RED, (size - size//4, size//1.5), size//3)
            pygame.draw.line(self.image, (0, 255, 0), (size//2, 0), (size//4, size//1.5), 2)
            pygame.draw.line(self.image, (0, 255, 0), (size//2, 0), (size - size//4, size//1.5), 2)
        else:
            pygame.draw.circle(self.image, PINK, (size//2, size//2), size//2) 
        
        self.rect = self.image.get_rect()
        self.rect.center = (x * TILE_SIZE + TILE_SIZE // 2, y * TILE_SIZE + TILE_SIZE // 2)

def draw_hearts(surface, x, y, lives):
    for i in range(lives):
        # Draw a heart
        # Heart shape using two circles and a triangle
        heart_color = RED
        offset_x = x - (i * 30) # Draw from right to left or define standard pos? 
        # Let's draw at x + i*spacing
        h_x = x + i * 35
        h_y = y
        
        # Left circle
        pygame.draw.circle(surface, heart_color, (h_x - 5, h_y - 5), 5)
        # Right circle
        pygame.draw.circle(surface, heart_color, (h_x + 5, h_y - 5), 5)
        # Triangle bottom
        points = [(h_x - 10, h_y - 2), (h_x + 10, h_y - 2), (h_x, h_y + 10)]
        pygame.draw.polygon(surface, heart_color, points)

class Joystick:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius
        self.knob_radius = radius // 2
        
    def draw(self, surface, direction):
        # Draw Base
        pygame.draw.circle(surface, (50, 50, 50), (self.x, self.y), self.radius)
        pygame.draw.circle(surface, (100, 100, 100), (self.x, self.y), self.radius, 2)
        
        # Calculate Knob Position
        dx, dy = direction
        knob_x = self.x + dx * (self.radius - self.knob_radius)
        knob_y = self.y + dy * (self.radius - self.knob_radius)
        
        # Draw Knob
        pygame.draw.circle(surface, (150, 150, 150), (int(knob_x), int(knob_y)), self.knob_radius)
        pygame.draw.circle(surface, WHITE, (int(knob_x), int(knob_y)), self.knob_radius, 2)

def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Pacman")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('arial', 20)
    game_over_font = pygame.font.SysFont('arial', 48)

    # Sprite Groups
    all_sprites = pygame.sprite.Group()
    walls = pygame.sprite.Group()
    pellets = pygame.sprite.Group()
    ghosts = pygame.sprite.Group()
    power_pellets = pygame.sprite.Group()
    
    player = None
    
    # Init Map
    for y, row in enumerate(GAME_MAP):
        for x, tile in enumerate(row):
            if tile == 1:
                wall = Wall(x, y)
                walls.add(wall)
                all_sprites.add(wall)
            elif tile == 0:
                pellet = Pellet(x, y, is_power=False)
                pellets.add(pellet)
                all_sprites.add(pellet)
            elif tile == 3:
                pellet = Pellet(x, y, is_power=True)
                power_pellets.add(pellet) # Add to both if want to count as pellet, or separate
                pellets.add(pellet) # Add to pellets to count for win condition
                all_sprites.add(pellet)
            elif tile == 9 or tile == 8:
                # Ghost House
                pass

    # Create Player - find a valid spot 
    player = Player(20, 23) # Shifted for 40 width
    all_sprites.add(player)

    # Create Ghosts - put in centered box (Row 14 is safe ghost house interior)
    ghost_colors = [RED, PINK, CYAN, ORANGE]
    # x=18, 19, 20, 21 in Row 14 are all '8' (Ghost House)
    ghost_start_positions = [(18, 14), (19, 14), (20, 14), (21, 14)] 
    for i in range(4):
        g_x, g_y = ghost_start_positions[i]
        ghost = Ghost(g_x, g_y, ghost_colors[i])
        ghosts.add(ghost)
        all_sprites.add(ghost)

    # UI Elements
    joystick = Joystick(100, SCREEN_HEIGHT - 35, 25) # Place on left side

    running = True
    game_state = "START" # START, PLAYING, GAMEOVER, WIN

    while running:
        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if game_state == "START":
                    game_state = "PLAYING"
                elif game_state == "PLAYING":
                    if event.key == pygame.K_LEFT:
                        player.next_direction = (-1, 0)
                    elif event.key == pygame.K_RIGHT:
                        player.next_direction = (1, 0)
                    elif event.key == pygame.K_UP:
                        player.next_direction = (0, -1)
                    elif event.key == pygame.K_DOWN:
                        player.next_direction = (0, 1)
                elif game_state in ["GAMEOVER", "WIN"]:
                    if event.key == pygame.K_r:
                        main() # Restart
                        return
            elif event.type == pygame.MOUSEBUTTONDOWN: # Start on click too
                if game_state == "START":
                     game_state = "PLAYING"

        if game_state == "PLAYING":
            # Updates
            player.update(walls)
            ghosts.update(walls)
            
            # Pellet Collisions
            hits = pygame.sprite.spritecollide(player, pellets, True)
            if hits:
                for hit in hits:
                     if hit.is_power:
                         player.score += 50
                         # Activate Power Mode
                         for ghost in ghosts:
                             ghost.scare()
                     else:
                         player.score += 10
                
                if len(pellets) == 0:
                    game_state = "WIN"

            # Ghost Collisions
            ghost_hits = pygame.sprite.spritecollide(player, ghosts, False)
            if ghost_hits:
                for ghost in ghost_hits:
                    if ghost.frightened:
                        # Eat Ghost
                        ghost.reset() # Send back to home
                        player.score += 200
                    else:
                        player.lives -= 1
                        if player.lives > 0:
                            player.reset()
                            for g in ghosts:
                                g.reset()
                            pygame.time.delay(1000) # Pause briefly
                            # Flush events to prevent buffered updates
                            pygame.event.clear() 
                        else:
                            game_state = "GAMEOVER"

        # Drawing
        screen.fill(BLACK)
        
        if game_state == "START":
            # Draw Title Screen
            title_text = game_over_font.render("PACMAN", True, YELLOW)
            start_text = font.render("Press Any Key to Start", True, WHITE)
            screen.blit(title_text, (SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2 - 50))
            screen.blit(start_text, (SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2 + 20))
            
            # Draw some demo sprites for fun
            demo_pacman = Player(SCREEN_WIDTH//TILE_SIZE//2 - 2, SCREEN_HEIGHT//TILE_SIZE//2 + 4)
            demo_pacman.image = pygame.transform.scale(demo_pacman.image, (60, 60))
            screen.blit(demo_pacman.image, (SCREEN_WIDTH//2 - 30, SCREEN_HEIGHT//2 - 150))

        elif game_state == "PLAYING" or game_state == "GAMEOVER" or game_state == "WIN":
            all_sprites.draw(screen)
            
            # Draw Score and Lives
            score_text = font.render(f"Score: {player.score}", True, WHITE)
            # Adjusted Score position to avoid joystick overlap
            screen.blit(score_text, (150, SCREEN_HEIGHT - 35))
            
            # Draw Joystick
            # Use next_direction if active, else current direction for responsiveness
            joy_dir = player.next_direction if player.next_direction != (0,0) else player.direction
            # Actually, pure direction is what's happening. Next direction is 'queued'.
            # Mimic action... let's just show current moving direction.
            joystick.draw(screen, player.direction)

            # Draw Hearts for Lives
            draw_hearts(screen, SCREEN_WIDTH - 120, SCREEN_HEIGHT - 25, player.lives)
            
            if game_state == "GAMEOVER":
                go_text = game_over_font.render("GAME OVER", True, RED)
                restart_text = font.render("Press R to Restart", True, WHITE)
                screen.blit(go_text, (SCREEN_WIDTH//2 - 140, SCREEN_HEIGHT//2 - 50))
                screen.blit(restart_text, (SCREEN_WIDTH//2 - 80, SCREEN_HEIGHT//2 + 10))
                
            elif game_state == "WIN":
                win_text = game_over_font.render("YOU WIN!", True, YELLOW)
                restart_text = font.render("Press R to Restart", True, WHITE)
                screen.blit(win_text, (SCREEN_WIDTH//2 - 120, SCREEN_HEIGHT//2 - 50))
                screen.blit(restart_text, (SCREEN_WIDTH//2 - 80, SCREEN_HEIGHT//2 + 10))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
