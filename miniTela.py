import pygame
import numpy as np
from PIL import Image
import pygame.draw_py
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

pygame.init()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


logical_width = 28
logical_height = 28
pixel_size = 20 

pygame.font.init()
font = pygame.font.SysFont(None, 20)
screen = pygame.display.set_mode((logical_width * pixel_size, logical_height * pixel_size))
pygame.display.set_caption("Paint Board")

screen.fill(BLACK)
pygame.display.flip()

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

def capture_and_resize():
    screen_array = pygame.surfarray.array3d(screen)
    screen_array = np.transpose(screen_array, (1, 0, 2))
    image = Image.fromarray(screen_array)
    image = image.convert("L")
    resized_image = image.resize((28, 28), Image.LANCZOS)
    resized_array = np.array(resized_image)
    return resized_array

drawSize = 5
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_DELETE:
                pygame.draw.rect(screen, BLACK, pygame.Rect(0, 0, 800, 600))
            elif event.key == pygame.K_c:
                resized_array = capture_and_resize()/255.0
                resized_array = np.expand_dims(resized_array, axis=0)
                prediction = model.predict(resized_array)
                for i, prob in enumerate(prediction[0]):
                    percent_prob = prob * 100
                    print(f'{class_names[i]}: {percent_prob:.2f}%')
            elif event.key == pygame.K_1:
                drawSize += 1
            elif event.key == pygame.K_2:
                drawSize -= 1
                if drawSize < 1 :
                    drawSize = 1
                
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
            if drawing:
                pygame.draw.circle(screen, WHITE, (mouse_x, mouse_y), drawSize)        
            elif erasing:
                pygame.draw.circle(screen, BLACK, (mouse_x, mouse_y), drawSize)
        pygame.display.flip()
    

pygame.quit()
