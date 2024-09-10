import pygame
import numpy as np
from PIL import Image
import pygame.draw_py
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time

pygame.init()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


logical_width = 28
logical_height = 28
pixel_size = 20 

pygame.font.init()
font = pygame.font.SysFont(None, 20)
screen = pygame.display.set_mode((logical_width * pixel_size, logical_height * pixel_size + 300))
pygame.display.set_caption("AI Board")

screen.fill(BLACK)
pygame.display.flip()

# Superfície separada para o desenho
draw_surface = pygame.Surface((logical_width * pixel_size, logical_height * pixel_size))
draw_surface.fill(BLACK)

running = True
drawing = False
erasing = False

data = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
train_images = train_images/255.0
test_images = test_images/255.0
model = keras.Sequential([
                    keras.layers.Flatten(input_shape = (28,28)),
                    keras.layers.Dense(128, activation = "relu"),
                    keras.layers.Dense(10, activation = "softmax")
                ])
model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
model.fit(train_images, train_labels, epochs = 5)

# Função para capturar a superfície de desenho e redimensionar
def capture_and_resize():
    draw_array = pygame.surfarray.array3d(draw_surface)
    draw_array = np.transpose(draw_array, (1, 0, 2))
    image = Image.fromarray(draw_array)
    image = image.convert("L")
    resized_image = image.resize((28, 28), Image.LANCZOS)
    resized_array = np.array(resized_image)
    return resized_array

def drawYAxis(max_height, y_start, x_pos):
    PINK = (255, 192, 203)
    font = pygame.font.SysFont(None, 24)

    for i in range(0, 101, 10): 
        y_pos = y_start + (max_height - int(i / 100 * max_height))
        pygame.draw.line(screen, PINK, (x_pos, y_pos), (x_pos + 10, y_pos), 2)
        label = font.render(str(i), True, PINK)
        screen.blit(label, (x_pos - 30, y_pos - 10))

def downPart(res):
    pygame.draw.rect(screen, (128, 128, 128), pygame.Rect(0, 600, 800, 300))
    res.sort(key=lambda x: x[1], reverse=True)
    blue = (0, 0, 255)
    
    start_x = 100
    start_y = 625
    bar_width = 30
    max_bar_height = 200

    drawYAxis(max_bar_height, start_y, start_x - 20)

    for ix, (digit, prob) in enumerate(res):
        bar_height = int(prob / 100 * max_bar_height) 
        pygame.draw.rect(screen, blue, (start_x + ix * (bar_width + 10), start_y + (max_bar_height - bar_height), bar_width, bar_height))
        
        font = pygame.font.SysFont(None, 36)
        digit_text = font.render(str(digit), True, (0, 0, 0))
        screen.blit(digit_text, (start_x + ix * (bar_width + 10) + bar_width // 4, start_y + max_bar_height + 10))

drawSize = 25
lista = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0)]
c = 0
while running:
    c += 1
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_DELETE:
                draw_surface.fill(BLACK)  # Limpa a área de desenho
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            if event.button == 1:
                drawing = True
            elif event.button == 3:
                erasing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                drawing = False
            elif event.button == 3:
                erasing = False
        elif event.type == pygame.MOUSEMOTION:
            mouse_x, mouse_y = event.pos
            if mouse_y < logical_height * pixel_size:  # Limita o desenho à área correta
                if drawing:
                    pygame.draw.circle(draw_surface, WHITE, (mouse_x, mouse_y), drawSize)        
                elif erasing:
                    pygame.draw.circle(draw_surface, BLACK, (mouse_x, mouse_y), drawSize)
                
        if c % 70 == 0:
            resized_array = capture_and_resize()/255.0
            resized_array = np.expand_dims(resized_array, axis=0)
            prediction = model.predict(resized_array)
            lista.clear()

            for i, prob in enumerate(prediction[0]):
                percent_prob = prob * 100
                lista.append((class_names[i], round(percent_prob, 2)))

        # Desenha a área de desenho e as barras de probabilidade
        screen.blit(draw_surface, (0, 0))  # Mostra a área de desenho
        downPart(lista)
        pygame.display.flip()

pygame.quit()
