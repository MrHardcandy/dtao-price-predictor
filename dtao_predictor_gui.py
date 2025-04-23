   #!/usr/bin/env python3
   # GUI
   import tkinter as tk
   from src.gui import DTAOPredictorGUI

   def main():
       """GUI"""
       root = tk.Tk()
       root.title("dTAO Price Predictor")
       app = DTAOPredictorGUI(root)
       root.mainloop()

   if __name__ == "__main__":
       main()
