tag = intentsList[0]["intent"]
    listOfIntents = intentsJson["intents"]
    for i in listOfIntents:
        if i["tag"] == tag:
            result = random.choice(i["response"])
            break
