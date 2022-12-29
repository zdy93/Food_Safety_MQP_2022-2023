from email.mime.text import MIMEText
import smtplib

class send_eamil(object):
    def __init__ (self ,account,password):
        """
        Gamil("zsp007","xxxx")
        """
        self .account=account
        self .password=password

    def send (self ,to,title,content):
        """
        send('zsp007@gmail.com,zsp747@gmail.com")
        """
        server = smtplib.SMTP('smtp.163.com',25) # host 
        server.starttls()
        server.login(self .account,self .password)

        msg = MIMEText(content)
        msg['Content-Type'] = 'text/plain; charset="utf-8"'
        msg['Subject'] = title
        msg['From'] = self .account
        msg['To'] = to
        server.sendmail(self .account, to ,msg.as_string())
        server.close()


