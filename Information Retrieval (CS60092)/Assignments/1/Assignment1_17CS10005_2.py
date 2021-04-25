from bs4 import BeautifulSoup as BS
import os
import re
import json


def chech_required_dirs():
    try:
        os.mkdir('./ECTText/')
    except OSError as e:
        print("Directory already Exists")

    try:
        os.mkdir('./ECTNestedDict/')
    except OSError as e:
        print("Directory already Exists")


def get_nested_dict(data_dir):
    
    count=0
    for file in os.listdir(data_dir):
        nestedDict = {}
        # nestedDict['Participants'] = []
        # nestedDict['pres'] = {}
        # nestedDict['Questionnaire'] = {}

        f=open(os.path.join(data_dir,file), encoding='utf8')
        soup = BS(f, "html.parser")

        firstpara = soup.find('p')
        date_regex = re.compile('(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2}),\s+(\d{4})')
        
        for date in re.finditer(date_regex, firstpara.text.strip()):
            nestedDict['Date'] = date.group()
            break

        
        text_corpus = ""
        case=-1

        participants = []
        presentations = []
        questionnaire = []

        QSpeaker = ""
        presentationValue = ""
        presentor = ""
        QRemark = ""
        date=""
        fg=1
        boolean=True
        
        for para in soup.find_all('p'):
            text_corpus+=para.text.strip()+" "
            boolean=True
            strong = para.find('strong')

            if strong:
                title = strong.getText()
                boolean=False

                if "Company Participants" in title:
                    case=0
                    fg=2
                    boolean=True
                    continue

                elif "Conference Call Participants" in title:
                    case=0
                    fg=2
                    boolean=True
                    continue

                elif "Question-and-Answer" in title:
                    if(presentor != "" and presentationValue != ""):
                        pres = {}
                        pres[presentor] = presentationValue
                        presentations.append(pres)
                        fg=3
                        boolean=True
                    else:
                        boolean=False
                    case = 1

                elif case != 1:
                    if(presentor != "" and presentationValue != ""):
                        pres = {}
                        pres[presentor] = presentationValue
                        presentations.append(pres)
                        fg=4
                    else:
                        boolean=True
                        fg=5
                    presentor  = title.strip()
                    presentationValue = ""
                    case = 2

                elif case == 1:
                    if(QRemark != "" and QSpeaker != ""):
                        QA = {}
                        QA['Speaker'], QA['Remark'] = QSpeaker, QRemark
                        questionnaire.append(QA)
                    else:
                        boolean=True
                        fg=2
                    QSpeaker = title.strip()
                    fg=5
                    QRemark = ""

            else:
                if case == 0:
                    participants.append(para.text.strip())
                
                elif case == 1:
                    if QRemark != "":
                        QRemark = QRemark + ' ' + para.text.strip()
                    else :
                        QRemark = para.text.strip()

                elif case == 2:
                    if(presentationValue != ""):
                        presentationValue = presentationValue + " " + para.text.strip();
                    elif fg==10 and boolean:
                        break    
                    else:
                        presentationValue = para.text.strip()
                    

        if QSpeaker != "" and QRemark != "":
            QA = {}
            QA['Speaker'] = QSpeaker
            QA['Remark'] = QRemark
            questionnaire.append(QA)
            fg=3
            boolean=True

        nestedDict['Participants'] = participants
        nestedDict['Presentation'] = presentations
        nestedDict['Questionnaire'] = questionnaire

        # print(nestedDict)
        newFile = open('./ECTNestedDict/'+file[:-4]+'txt', "w")
        newFile.write(json.dumps(nestedDict))
        textFile = open('./ECTText/'+file[:-4]+'txt', "w")
        try:
            textFile.write(text_corpus)
        except:
            print("write Error in ", count)

        print("file ID: " , count, "----> Completed")
        count += 1

def main():
    data_dir="ECT"

    chech_required_dirs()
    get_nested_dict(data_dir)

if __name__=='__main__':
    main()