import os
import argparse
import tkinter as tk

import torch
import matplotlib.pyplot as plt

from src.models import MLP, CNN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILL_MAIN = 1 
FILL_PAD = .4


class MainWindow:
    """This class defines a window application using tkinter that
    will allow users to draw digits and pass the resulting image to
    a model to predict the draw digit.
    """
    def __init__(self, model_filename, model, grid_size=28, canvas_size=400):
        """Defines a MainWindow with a given grid_size, which will be a squared to
        produce a grid. This is going to be 28x28 to be consistent with MNIST
        unless the user specifies otherwise.
        """
        self.root = tk.Tk()
        self.root.title("MNIST UI")
        self.grid_size = grid_size
        self.canvas_size = canvas_size
        self.cell_size = canvas_size // grid_size
                
        self.grid_state = torch.full((self.grid_size, self.grid_size), 0)
        self.model = model
        self.model.load_state_dict(torch.load(os.path.join(BASE_DIR, f'models/{model_filename}')))

        self.canvas = tk.Canvas(self.root, bg="white", width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack(expand=tk.NO, fill=tk.BOTH)
        self.canvas.bind("<B1-Motion>", self.paint)
        predict_button = tk.Button(self.root, text="Predict", command=self.model_predict)
        predict_button.place(x=0, y=0)


    def paint(self, event):
        """Paints a given cell in the window grid. Black only for now..."""
        x, y = event.x, event.y
        grid_x, grid_y = x // self.cell_size, y // self.cell_size
        if grid_x < self.grid_size and grid_y < self.grid_size:
            self.grid_state[grid_y:grid_y+2, grid_x:grid_x+2] = FILL_MAIN
            self.canvas.create_rectangle(
                grid_x * self.cell_size,
                grid_y * self.cell_size,
                (grid_x + 2) * self.cell_size,
                (grid_y + 2) * self.cell_size,
                fill="black",
                outline="black",
            )
        
        ## This adds a grey shader to the drawn input - initially created to boost model predictions
        ## but turns out you can just build a better model to make better predictions who knew?
        # for i in [-1, 0, 1]:
        #     for j in [-1, 0, 1]:
        #         if ((0 <= grid_x + i < self.grid_size)
        #             and (0 <= grid_y + j < self.grid_size)
        #             and self.grid_state[grid_y + j][grid_x + i] != FILL_MAIN):
        #             # if we are in a valid grid that has not yet been rendered then we pad with gray.
        #             self.grid_state[grid_y + j][grid_x + i] = FILL_PAD
        #             self.canvas.create_rectangle(
        #                 (grid_x + i) * self.cell_size,
        #                 (grid_y + j) * self.cell_size,
        #                 (grid_x + i + 1) * self.cell_size,
        #                 (grid_y + j + 1) * self.cell_size,
        #                 fill="gray90",
        #                 outline="gray90",
        #             )

    def model_predict(self):
        """Consumes the current grid state and returns a prediction for the drawn number."""
        grid_state_norm = (self.grid_state - 0.1307) / 0.3081
        output = self.model(grid_state_norm.reshape(1,-1)).argmax(dim=1)
        self.canvas.delete("all")
        self.grid_state = torch.full((self.grid_size, self.grid_size), 0) 
        print(output)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--filename',
        '-f',
        type=str,
        help='Name of model artifact in the `models` directory to deploy for inference.',
    )
    parser.add_argument(
        '--model',
        '-m',
        type=str,
        help="Type of model being loaded. Must be one of 'MLP' or 'CNN'"
    )
    args = parser.parse_args()

    if args.model.upper() not in ['MLP','CNN']:
        raise ValueError("--model must be one of 'MLP', or 'CNN'")

    model = MLP() if args.model.upper() == 'MLP' else CNN()
    window = MainWindow(model_filename=args.filename, model=model)
    window.root.mainloop()
