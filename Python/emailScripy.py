#! /usr/local/bin/python



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