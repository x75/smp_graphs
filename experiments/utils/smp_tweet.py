"""smp_tweet

Tweet an image and text given on the command line
"""

import argparse, os, sys
import tweepy

def post_tweet_image(api, status, image):
    print("Pushing img = '%s', status = '%s'" %( args.image, args.status))
    api.update_with_media(args.image, args.status)

# def post_tweet_paper(api, status, image):
#     print "Pushing img = '%s', status = '%s'" %( args.image, args.status)
#     api.update_with_media(args.image, args.status)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image',  type = str, help = 'Path to image to use in the tweet', default = None)
    parser.add_argument('-s', '--status', type = str, help = 'Status text to tweet', default = '')
    args = parser.parse_args()

    # app credentials
    """this file contains four lines with
    consumer_key = "fdlgjdflgjdflkg"
    consumer_secret = "dslkfjsdlfkjd"
    access_token = "sldkfjlskdfjsdklf"
    access_token_secret = "dfkjsldkfj"
    """
    f = open('utils/smp_tweet_cred.py', 'r')
    code = compile("".join(f.readlines()), "<string>", "exec")
    # prepare variables
    global_vars = {}
    local_vars  = {}
    # run the code
    exec(code, global_vars, local_vars)

    # auth and connect to api
    auth = tweepy.OAuthHandler(local_vars['consumer_key'], local_vars['consumer_secret'])
    auth.set_access_token(local_vars['access_token'], local_vars['access_token_secret'])
    api = tweepy.API(auth)

    # public_tweets = api.home_timeline()
    # for tweet in public_tweets:
    #     print tweet.text

    # api.update_status('x')
    
    # img = 'temp_c.png'
    # status = 'temp_c.png'

    # img = 'inverted/plot_nodes_over_data_scattermatrix_hexbin_smpHebbianSOM.jpg'
    # status = 'hebbian connected SOMs with 3K timesteps of online learning on inverted sine, #smp_base'

    post_tweet_image(api, args.status, args.image)
