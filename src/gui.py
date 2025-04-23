def create_price_prediction_tab(self):
    """Create price prediction tab"""
    tab = ttk.Frame(self.notebook)
    self.notebook.add(tab, text=self.i18n.get_text("tab_price_prediction", "Price Prediction"))
    
    # Control panel
    control_frame = ttk.LabelFrame(tab, text=self.i18n.get_text("options", "Options"))
    control_frame.pack(fill=tk.X, padx=10, pady=5)
    
    # Subnet selection
    tk.Label(control_frame, text=self.i18n.get_text("select_subnet", "Select Subnet:")).pack(side=tk.LEFT, padx=5)
    self.prediction_subnet_var = tk.StringVar()
    self.prediction_subnet_selector = ttk.Combobox(control_frame, textvariable=self.prediction_subnet_var, width=10)
    self.prediction_subnet_selector.pack(side=tk.LEFT, padx=5)
    
    # Prediction days
    tk.Label(control_frame, text=self.i18n.get_text("prediction_days", "Days:")).pack(side=tk.LEFT, padx=5)
    self.prediction_days_var = tk.IntVar(value=30)
    days_spinbox = tk.Spinbox(control_frame, from_=7, to=365, textvariable=self.prediction_days_var, width=5)
    days_spinbox.pack(side=tk.LEFT, padx=5)
    
    # Model selection
    tk.Label(control_frame, text=self.i18n.get_text("prediction_model", "Model:")).pack(side=tk.LEFT, padx=5)
    self.prediction_model_var = tk.StringVar(value="random_forest")
    models = [
        ("random_forest", "Random Forest"),
        ("linear", "Linear Regression"),
        ("svr", "SVR"),
        ("lstm", "LSTM"),
        ("arima", "ARIMA"),
        ("xgboost", "XGBoost"),
        ("prophet", "Prophet")
    ]
    model_selector = ttk.Combobox(control_frame, textvariable=self.prediction_model_var, 
                                 values=[m[0] for m in models], state="readonly", width=15)
    model_selector.pack(side=tk.LEFT, padx=5) 