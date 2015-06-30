#! /usr/local/bin/python
"""
Handy script that can send email alerts to yourself (or another for that matter). This allows for alerts as to
progress of the code, or in the case of failures, no alerts at all.
"""



def alertMe(message):
    import smtplib
    # Stack Overflow assisted

    sender = 'nayyarv@gmail.com'
    receiver = 'nayyarv@gmail.com'

    msg = "\r\n".join([
        "From: nayyarv@gmail.com",
        "To: nayyarv@gmail.com",
        "Subject: Alert with computations",
        "",
        "Computations finished: "
    ])

    msg += str(message)
    username = 'nayyarv@gmail.com'
    password = 'mhtrynnwmfqdehlk'

    try:
        server = smtplib.SMTP('smtp.gmail.com:587')
        server.ehlo()
        server.starttls()
        server.login(username, password)
        server.sendmail(sender, receiver, msg)
        server.quit()
    except:
        print "Error occured :'("


if __name__ == "__main__":
    alertMe("Testing testing!!")