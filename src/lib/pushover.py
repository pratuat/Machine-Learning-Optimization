import http.client, urllib

def notify_me(message, priority = -1):
    conn = http.client.HTTPSConnection("api.pushover.net:443")
    conn.request("POST", "/1/messages.json",
      urllib.parse.urlencode({
        "token": "au4ft3tx933a6og2du5gbahu6tzmjv",
        "user": "uHdQmSoJL6Sggh3akTyZYpWwJFjeJ7",
        "message": message,
        "priority": priority
      }), { "Content-type": "application/x-www-form-urlencoded" })
    conn.getresponse()