import pygame
import math
import time
import numpy as np

class Sensor(object):
    def __init__(self,x,y,angle):
        self.x = x
        self.y = y
        self.angle = angle

    def connect(x1, y1, x2, y2):
        dx, dy = abs(x2-x1), abs(y2-y1)
        if dx > dy:
            n_steps = dx + 1
        else:
            n_steps = dy + 1
        
        x = np.linspace(x1, x2, n_steps, dtype=np.int32)
        y = np.linspace(y1, y2, n_steps, dtype=np.int32)
        return x, y
        
    def detect(map):
        pass

class Car(object):
    def __init__(self,x, y, yaw, width=50, height=100):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.width = width
        self.height = height
        self.L = 60.0
        self.steer_coef = 25./50.
        self.speed_coef = 1.0
        self.hitpoints = [pygame.math.Vector2(-50,25), pygame.math.Vector2(50,25), pygame.math.Vector2(50,-25), pygame.math.Vector2(-50,-25),
                          pygame.math.Vector2(0,25), pygame.math.Vector2(0,-25)]
        self.img = pygame.transform.scale(pygame.image.load("car.png"), (height, width))
        self.img = pygame.transform.rotate(self.img, 180.0)
        self.steer = 0

    def move(self, steer, speed, dt):
        speed = self.speed_coef * speed
        angle = -self.steer_coef * steer
        self.x += speed * math.cos(self.yaw) * dt
        self.y -= speed * math.sin(self.yaw) * dt
        self.yaw += speed * math.tan(math.radians(angle)) * dt / self.L
        self.yaw = self.yaw % (2*math.pi)
        rotated_offset = pygame.math.Vector2(30.0,0.0).rotate(-math.degrees(self.yaw))
        self.center = (self.x, self.y) + rotated_offset

    def rotate(self, angle):
        rotated_image = pygame.transform.rotate(self.img, angle)
        rect = rotated_image.get_rect(center=self.center)
        return rotated_image, rect

    def draw(self, win):
        img, rect = self.rotate(math.degrees(self.yaw))
        win.blit(img, rect)

    def collision_check(self, map):
        angle = math.degrees(self.yaw)
        rotated_hitpoints = [x.rotate(-angle) for x in self.hitpoints]
        for point in rotated_hitpoints:
            x, y = int(point[0]+self.center[0]) , int(point[1]+self.center[1])
            if map.img.get_at((x,y))[0] < 100:
                return True
            pygame.draw.circle(game.win, (255,0,0), (x,y), 10)
        return False

class Map:
    def __init__(self,):
        self.img = pygame.image.load("map.png")
    
    def draw(self, win):
        win.blit(self.img, (0,0))

class Game:
    def __init__(self, W, H, objects):
        pygame.init()
        pygame.display.set_caption("Car Game")
        self.win =  pygame.display.set_mode((W, H))
        self.run = True
        self.clock = pygame.time.Clock()
        self.objects = objects

    def draw(self):
        for obj in self.objects:
            obj.draw(self.win)

    def quit(self):
        pygame.quit()

car = Car(200,700,math.pi / 2)
map = Map()
game = Game(1482, 745, [map, car])

while game.run:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game.run = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                car.steer = -50
            if event.key == pygame.K_RIGHT:
                car.steer = 50
            if event.key == pygame.K_UP:
                car.steer = 0
            if event.key == pygame.K_r:
                car.x = 200
                car.y = 700
                car.yaw = math.pi / 2.0
                car.steer = 0
    
    car.move(car.steer, 50, 0.033)
    game.draw()
    pygame.display.update()
    if car.collision_check(map) :
        car.x = 200
        car.y = 700
        car.yaw = math.pi / 2.0
        car.steer = 0
    game.clock.tick(30)
    
game.quit()