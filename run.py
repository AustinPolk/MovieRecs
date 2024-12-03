import os
from chat import Chatter

if __name__ == '__main__':
    chatter = Chatter()
    os.system('cls')    # clear screen from initializing chatbot
    chatter.chat(debug=False)