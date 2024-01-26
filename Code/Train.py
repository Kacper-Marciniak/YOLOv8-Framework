"""
Train YOLOv8 object detection model
"""

import os
from utility.GUI import askdirectory, choosefromlist
from training.CTrainer import CTrainer
from parameters.TrainingParameters import AVAILABLE_MODELS
from path.root import ROOT_DIR

# Training epochs
i_Epochs = 25
f_ConfThreshTest = 0.5
i_BatchSize = 16

if __name__ == "__main__":
    # Choose model type
    s_ModelName = choosefromlist(AVAILABLE_MODELS, title="Choose model size", width=75)
    # Set dataset directory
    s_DatasetDir = askdirectory("Select dataset folder", initialdir=os.path.join(ROOT_DIR,'datasets'))

    try:
        # Initialize traine mr class
        c_Trainer = CTrainer(
            s_ModelName = s_ModelName,
            s_DatasetDirectory = s_DatasetDir
        )
        # Print trainer info
        c_Trainer.PrintInfo()
        # Train model
        c_Trainer.Train(i_Epochs, i_BatchSize)
        # Copy data configuration
        c_Trainer._CopyDataConfig()
        # Run test inference
        c_Trainer.TestInference(f_ConfThreshTest)
        # Visualize training results
        c_Trainer.PlotResults()

        print("All tasks finished!")
    except Exception as e:
        print(f"\n\nTraining stopped. Error message:\n{e}")