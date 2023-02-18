# Keep_Babies_Safe

Configuration: Chip i5-1135G7 2,4 GHz, Ram 8GB, Window    
Using virtualenv (or conda) python >= virtualenv 3.7.1 

Step 1: Open cmd on your computer and git clone https://github.com/nghia-cx/Keep_Babies_Safe.git

Step 1: Open virtualenv/conda on your computer

Step 2: Go to repository which you just cloned to your computer

Step 3: pip install -r requirements.txt

Step 4: You can add the function of sending notifications to your phone via telegram or not  

If you want add, you must get API Token Telegram (follow https://www.siteguarding.com/en/how-to-get-telegram-bot-api-token?fbclid=IwAR2Z0Yz2c2rLCOi6eA8BxjS2KB7flyR3C2Yil0nvo2Rg29GUrbXcZLq6Xbc)    
Open send_telegram.py:  
    At line 5: replace your_token  
    At line 6: replace your_id  

Step 5: Run main.py
    It has 2 mode: Indoor and Outdoor. Press 1 for Indoor, 2 for Outdoor  
    In mode 1:
        When the video start, you can choose points via 'left click mouse'. After choose, press button D ...  
    In mode 2:
        When the video start, press button D ...  
    For all mode, press button:  
     P: it will pause your video,  
     Q: it will quit your video,  
     D: it will detect object on your video,  
     C: it will not detect object on your video  
    
