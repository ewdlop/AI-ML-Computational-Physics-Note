import random
import re

bot_template = "BOT: {0}"
user_template = "USER: {0}"

rules = {
    "i want (.*)" : ['What would it mean if you got {0}', "Why do you want {0}", "What's stopping you from getting {0}"],
    "do you remember (.*)" : ["Did you think I would forget {0}","Why haven't you been able to forget {0}","What about {0}","Yes...and?"],
    "do you believe in (.*)" : ["I don't believe in {0}" , "I only believe in myself"],
    "do you think (.*)" : ['if {0}? Absoultely.','No chance'],
    "if (.*)": ["Do you really think it's likely that {0}","Do you wish that {0}","What do you think about {0}", "Really--if{0}"],
    "do you like (.*)" : ['I like nothing'],
    "my name is (.*)" : ['{0} is a beatiful name']
}

none_sense = {
    "default" : ["I like chicken", "I dont know what I am talking about","Weather is good", "I am drunk"]
}

def response(message):
    for pattern, responses in rules.items():
        match = re.search(pattern, message.lower())
        if(match is not None):
            n = random.randint(0, len(responses)-1)
            if('{0}' in responses[n]):
                phrase  = match.group(1)
                responses[n] = responses[n].format(swap_pronouns(phrase))
                return responses[n]
            else:
                return responses[n]
    #no match 
    n = random.randint(0, len(none_sense['default'])-1)
    return none_sense['default'][n]

def swap_pronouns(message):
    if 'me' in message:
        return re.sub('me','you', message)
    if 'my' in message:
        return re.sub('my','your', message)
    if 'your' in message:
        return re.sub('your','my', message)
    if 'you' in message:
        return re.sub('you','me', message)
    else:
        return message
def send_message(message):
    print(user_template.format(message))
    bot_response = response(message)
    print(bot_template.format(bot_response))

print(swap_pronouns("You hate my dog "))
send_message("do you remember your name?")
send_message("I want a drink")
send_message("do you believe in god")
send_message("do you like candy")
send_message("hello")