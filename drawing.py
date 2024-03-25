import pygame
import numpy as np

def get_drawing_matrix():
    # Initialize pygame
    pygame.init()

    # Set the dimensions of the window and the drawing area
    window_width = 280
    window_height = 280
    drawing_width = 28
    drawing_height = 28

    # Create the window
    window = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Drawing Program")

    # Create a 2D matrix to store the drawing
    drawing_matrix = np.zeros((drawing_height, drawing_width), dtype=np.uint8)

    # Set the drawing color to white
    drawing_color = (255, 255, 255)

    # Main loop
    running = True
    drawing = False  # Flag to indicate if the mouse button is pressed and held
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True  # Start drawing
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False  # Stop drawing
            elif event.type == pygame.MOUSEMOTION and drawing:
                # Get the position of the mouse
                x, y = pygame.mouse.get_pos()

                # Calculate the corresponding cell in the drawing matrix
                cell_x = x // (window_width // drawing_width)
                cell_y = y // (window_height // drawing_height)

                # Set the corresponding cell in the drawing matrix to 255
                drawing_matrix[cell_y][cell_x] = 255

        # Clear the window
        window.fill((0, 0, 0))

        # Draw the drawing matrix on the window
        for y in range(drawing_height):
            for x in range(drawing_width):
                if drawing_matrix[y][x] > 0:
                    pygame.draw.rect(window, drawing_color, (x * (window_width // drawing_width), y * (window_height // drawing_height), window_width // drawing_width, window_height // drawing_height))

        # Update the window
        pygame.display.flip()

    # Save the drawing matrix as a text file
    np.savetxt("drawing_matrix.txt", drawing_matrix, fmt="%d")

    # Quit pygame
    pygame.quit()

    # Return the drawing matrix
    return drawing_matrix
