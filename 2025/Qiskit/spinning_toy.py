from psychopy import visual, core, event
import numpy as np

# Create a PsychoPy window
win = visual.Window(size=(800, 600), color=(0, 0, 0))

# Load toy image (replace with your toy image)
toy = visual.ImageStim(win, image="toy.png", size=(0.3, 0.3))

# Animation variables
angle = 0
t = 0

# Start animation loop
while True:
    # Floating effect (sinusoidal motion)
    y_pos = 0.3 * np.sin(2 * np.pi * t / 100)

    # Spinning effect
    angle += 2
    toy.setOri(angle)
    toy.setPos((0, y_pos))
    
    # Draw toy
    toy.draw()
    win.flip()

    # Time update
    t += 1
    core.wait(0.01)

    # Exit on key press
    if event.getKeys():
        break

win.close()
