#this module will send emails.
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def SendEmail(To="To Email address",message="This is the body of the Mail"):
    MY_ADDRESS = 'upgradchatbotassignment2914@gmail.com'
    PASSWORD = 'ttggvvyyhhbbuujjnn'

    msg = MIMEMultipart()       # create a message

    ## setup the parameters of the message
    msg['From']=MY_ADDRESS
    msg['To']=To
    msg['Subject']="Your Restaurants List"

    try:        
        # add in the message body
        msg.attach(MIMEText(message, 'plain'))
        # creates SMTP session 
        s = smtplib.SMTP('smtp.gmail.com', 587) 
      
        # start TLS for security 
        s.starttls() 
      
        # Authentication 
        s.login(MY_ADDRESS, PASSWORD) 
      
        # sending the mail 
        s.send_message(msg)

    except Exception as ex:
        print(ex)
        print("Unable to send the Email. Please try again later.")
        return  False
    finally:
        # terminating the session 
        s.quit()
    return True

if __name__=="__main__":
    SendEmail("dknayakbu@gmail.com","mail body")
