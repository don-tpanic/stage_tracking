# stage_tracking
Sample scripts for chatbot diaglog stage tracking and prediction. 

### Problem statement
To keep track of the stage of a user-assistant dialog based on standardised SOP and to predict the incoming stage for the assistant to better respond to the user. 

### Current solutions
1. Text classification (`run_text_cls.py`) - fine-tuning LMs to predict the current/future stage of a dialog given chat history.
2. Constrained generation (`run_constrained_gen.py`) - fine-tuning LMs to generate in natural language the current/future stage of a given dialog. 
