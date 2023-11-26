import cv2
import os
import pandas as pd
import sys
import speech_recognition as sr
from gtts import gTTS
import platform
from pandasai.llm import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import Chroma
import constants
os.environ["OPENAI_API_KEY"] = constants.APIKEY


llm = OpenAI(api_token="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")


def con_typing():
    PERSIST = False

    query = None
    if len(sys.argv) > 1:
        query = sys.argv[1]

    data_directory = "/home/inokov/PycharmProjects/Assistant/data"

    if PERSIST and os.path.exists("persist"):
        print("Reusing index...\n")
        vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        # Use the absolute path to the 'data/' directory
        loader = DirectoryLoader(data_directory)

        if PERSIST:
            index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader])
        else:
            index = VectorstoreIndexCreator().from_loaders([loader])

    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
    )

    chat_history = []
    while True:
        if not query:
            query = input("Prompt: ")
        if query in ['quit', 'q', 'exit']:
            sys.exit()
        result = chain({"question": query, "chat_history": chat_history})
        print(result['answer'])

        chat_history.append((query, result['answer']))
        query = None

def voice_to_text():
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        print("Prompt:")
        audio = recognizer.listen(source, timeout=10)

        try:
            # Use Google Web Speech API to convert speech to text
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            print("Sorry, I could not understand what you said.")
        except sr.RequestError as e:
            print(f"Could not request results from Google Web Speech API; {e}")

def text_to_speech(text, language='en', output_file='output.mp3'):
    # Create a gTTS object
    tts = gTTS(text=text, lang=language, slow=False)

    # Save the speech as an MP3 file
    tts.save(output_file)

    # Play the generated speech
    if platform.system() == "Windows":
        os.system(f"start {output_file}")
    elif platform.system() == "Linux" or platform.system() == "Darwin":  # Unix-like system
        os.system(f"xdg-open {output_file}")
    else:
        print("Unsupported operating system. Please open the audio file manually.")
def con_voice():
    PERSIST = False

    query = None
    if len(sys.argv) > 1:
        query = sys.argv[1]

    data_directory = "/home/inokov/PycharmProjects/Assistant/data"

    if PERSIST and os.path.exists("persist"):
        print("Reusing index...\n")
        vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        # Use the absolute path to the 'data/' directory
        loader = DirectoryLoader(data_directory)

        if PERSIST:
            index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader])
        else:
            index = VectorstoreIndexCreator().from_loaders([loader])

    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
    )

    chat_history = []
    while True:
        if not query:
            query = voice_to_text()
        if query in ['quit', 'q', 'exit']:
            sys.exit()
        result = chain({"question": query, "chat_history": chat_history})
        text_to_speech(result['answer'])

        chat_history.append((query, result['answer']))
        query = None

def show_all(pid):
    print(pid)
    df = pd.read_csv("Database_of_Supermarket.csv", index_col='Product ID')
    print(df.loc[pid])

def qr_code():
    # Set the environment variable to use XCB for QT applications
    os.environ['QT_QPA_PLATFORM'] = 'xcb'

    cap = cv2.VideoCapture(0)
    detector = cv2.QRCodeDetector()

    while True:
        _, img = cap.read()

        data, bbox, _ = detector.detectAndDecode(img)

        # check if there is a QRCode in the image
        if data:
            print("QR Code data:", data)
            return show_all(data)

        cv2.imshow("QRCODEscanner", img)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def main_en():
    print("Hello this is shopping Assistant. Please choose options below")
    print("1. QR  code scanning")
    print("2. Conversation via typing")
    print("3. Conversation via voice")
    print("4. Change language")
    print("5. Exit")
    choice = int(input("Your choice: "))

    if choice == 1:
        qr_code()
    elif choice == 2:
        con_typing()
    elif choice == 3:
        con_voice()
    elif choice == 4:
        lan_en()
    elif choice == 5:
        return 0
    else:
        print("Wrong input. Please, try again")
        main_en()

def lan_en():
    print("Please choose a language")
    print("1. English")
    print("2. русский язык")
    print("3. 한국어")
    lang = int(input("Your choice: "))
    if lang == 1:
        main_en()
    elif lang == 2:
        main_ru()
    elif lang == 3:
        main_kr()
    else:
        print("Wrong input. Please try again")
        lan_en()

def main_ru():
    print("Здравствуйте, это помощник по покупкам.  Пожалуйста, выберите варианты ниже")
    print("1. Сканирование QR-кода")
    print("2. Разговор посредством набора текста")
    print("3. Разговор посредством голоса")
    print("4. Изменить язык")
    choice = int(input("твой выбор: "))
    if choice == 1:
        qr_code()
    elif choice == 2:
        con_typing()
    elif choice == 3:
        con_voice()
    elif choice == 4:
        lan_en()
    elif choice == 5:
        return 0
    else:
        print("Неправильный ввод. Пожалуйста, попробуйте еще раз")
        main_en()

def lan_ru():
    print("Пожалуйста, выберите язык")
    print("1. English")
    print("2. русский язык")
    print("3. 한국어")
    lang = int(input("твой выбор: "))
    if lang == 1:
        main_en()
    elif lang == 2:
        main_ru()
    elif lang == 3:
        main_kr()
    else:
        print("Неправильный ввод. Пожалуйста, попробуйте еще раз")
        lan_ru()

def main_kr():
    print("안녕하세요 쇼핑도우미 입니다.  아래 옵션을 선택해주세요")
    print("1. QR 코드 스캔")
    print("2. 타이핑을 통한 대화")
    print("3. 음성을 통한 대화")
    print("4. 언어 변경")
    choice = int(input("선택: "))
    if choice == 1:
        qr_code()
    elif choice == 2:
        con_typing()
    elif choice == 3:
        con_voice()
    elif choice == 4:
        lan_en()
    elif choice == 5:
        return 0
    else:
        print("입력이 잘못되었습니다. 다시 시도해 주세요")
        main_en()

def lan_kr():
    print("언어를 선택하세요")
    print("1. English")
    print("2. русский язык")
    print("3. 한국어")
    lang = int(input("선택: "))
    if lang == 1:
        main_en()
    elif lang == 2:
        main_ru()
    elif lang == 3:
        main_kr()
    else:
        print("입력이 잘못되었습니다. 다시 시도해 주세요")
        lan_en()

main_en()
