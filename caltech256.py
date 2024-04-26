from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
from itertools import product

# Example list of items
items = {
    "ak47": "001", "american-flag": "002", "backpack": "003", "baseball-bat": "004", "baseball-glove": "005",
    "basketball-hoop": "006", "bat": "007", "bathtub": "008", "bear": "009", "beer-mug": "010",
    "billiards": "011", "binoculars": "012", "birdbath": "013", "blimp": "014", "bonsai-101": "015",
    "boom-box": "016", "bowling-ball": "017", "bowling-pin": "018", "boxing-glove": "019", "brain-101": "020",
    "breadmaker": "021", "buddha-101": "022", "bulldozer": "023", "butterfly": "024", "cactus": "025",
    "cake": "026", "calculator": "027", "camel": "028", "cannon": "029", "canoe": "030",
    "car-tire": "031", "cartman": "032", "cd": "033", "centipede": "034", "cereal-box": "035",
    "chandelier-101": "036", "chess-board": "037", "chimp": "038", "chopsticks": "039", "cockroach": "040",
    "coffee-mug": "041", "coffin": "042", "coin": "043", "comet": "044", "computer-keyboard": "045",
    "computer-monitor": "046", "computer-mouse": "047", "conch": "048", "cormorant": "049", "covered-wagon": "050",
    "cowboy-hat": "051", "crab-101": "052", "desk-globe": "053", "diamond-ring": "054", "dice": "055",
    "dog": "056", "dolphin-101": "057", "doorknob": "058", "drinking-straw": "059", "duck": "060",
    "dumb-bell": "061", "eiffel-tower": "062", "electric-guitar-101": "063", "elephant-101": "064", "elk": "065",
    "ewer-101": "066", "eyeglasses": "067", "fern": "068", "fighter-jet": "069", "fire-extinguisher": "070",
    "fire-hydrant": "071", "fire-truck": "072", "fireworks": "073", "flashlight": "074", "floppy-disk": "075",
    "football-helmet": "076", "french-horn": "077", "fried-egg": "078", "frisbee": "079", "frog": "080",
    "frying-pan": "081", "galaxy": "082", "gas-pump": "083", "giraffe": "084", "goat": "085",
    "golden-gate-bridge": "086", "goldfish": "087", "golf-ball": "088", "goose": "089", "gorilla": "090",
    "grand-piano-101": "091", "grapes": "092", "grasshopper": "093", "guitar-pick": "094", "hamburger": "095",
    "hammock": "096", "harmonica": "097", "harp": "098", "harpsichord": "099", "hawksbill-101": "100",
    "head-phones": "101", "helicopter-101": "102", "hibiscus": "103", "homer-simpson": "104", "horse": "105",
    "horseshoe-crab": "106", "hot-air-balloon": "107", "hot-dog": "108", "hot-tub": "109", "hourglass": "110",
    "house-fly": "111", "human-skeleton": "112", "hummingbird": "113", "ibis-101": "114", "ice-cream-cone": "115",
    "iguana": "116", "ipod": "117", "iris": "118", "jesus-christ": "119", "joy-stick": "120",
    "kangaroo-101": "121", "kayak": "122", "ketch-101": "123", "killer-whale": "124", "knife": "125",
    "ladder": "126", "laptop-101": "127", "lathe": "128", "leopards-101": "129", "license-plate": "130",
    "lightbulb": "131", "light-house": "132", "lightning": "133", "llama-101": "134", "mailbox": "135",
    "mandolin": "136", "mars": "137", "mattress": "138", "megaphone": "139", "menorah-101": "140",
    "microscope": "141", "microwave": "142", "minaret": "143", "minotaur": "144", "motorbikes-101": "145",
    "mountain-bike": "146", "mushroom": "147", "mussels": "148", "necktie": "149", "octopus": "150",
    "ostrich": "151", "owl": "152", "palm-pilot": "153", "palm-tree": "154", "paperclip": "155",
    "paper-shredder": "156", "pci-card": "157", "penguin": "158", "people": "159", "pez-dispenser": "160",
    "photocopier": "161", "picnic-table": "162", "playing-card": "163", "porcupine": "164", "pram": "165",
    "praying-mantis": "166", "pyramid": "167", "raccoon": "168", "radio-telescope": "169", "rainbow": "170",
    "refrigerator": "171", "revolver-101": "172", "rifle": "173", "rotary-phone": "174", "roulette-wheel": "175",
    "saddle": "176", "saturn": "177", "school-bus": "178", "scorpion-101": "179", "screwdriver": "180",
    "segway": "181", "self-propelled-lawn-mower": "182", "sextant": "183", "sheet-music": "184", "skateboard": "185",
    "skunk": "186", "skyscraper": "187", "smokestack": "188", "snail": "189", "snake": "190",
    "sneaker": "191", "snowmobile": "192", "soccer-ball": "193", "socks": "194", "soda-can": "195",
    "spaghetti": "196", "speed-boat": "197", "spider": "198", "spoon": "199", "stained-glass": "200",
    "starfish-101": "201", "steering-wheel": "202", "stirrups": "203", "sunflower-101": "204", "superman": "205",
    "sushi": "206", "swan": "207", "swiss-army-knife": "208", "sword": "209", "syringe": "210",
    "tambourine": "211", "teapot": "212", "teddy-bear": "213", "teepee": "214", "telephone-box": "215",
    "tennis-ball": "216", "tennis-court": "217", "tennis-racket": "218", "theodolite": "219", "toaster": "220",
    "tomato": "221", "tombstone": "222", "top-hat": "223", "touring-bike": "224", "tower-pisa": "225",
    "traffic-light": "226", "treadmill": "227", "triceratops": "228", "tricycle": "229", "trilobite-101": "230",
    "tripod": "231", "t-shirt": "232", "tuning-fork": "233", "tweezer": "234", "umbrella-101": "235",
    "unicorn": "236", "vcr": "237", "video-projector": "238", "washing-machine": "239", "watch-101": "240",
    "waterfall": "241", "watermelon": "242", "welding-mask": "243", "wheelbarrow": "244", "windmill": "245",
    "wine-bottle": "246", "xylophone": "247", "yarmulke": "248", "yo-yo": "249", "zebra": "250",
    "airplanes-101": "251", "ar-side-101": "252", "faces-easy-101": "253", "greyhound": "254", "tennis-shoes": "255",
    "toad": "256"
}

sims = {}

for word1 in items.keys():
    print(word1)
    w1 = word1[:]
    word1 = word1.replace('-101', '')
    word1 = re.sub("[^a-zA-Z]+", " ", word1)
    syns1 = wordnet.synsets(word1)
    for word2 in items.keys():
        if word1 != word2:
            w2 = word2[:]
            word2 = word2.replace('-101', '')
            word2 = re.sub("[^a-zA-Z]+", " ", word2)
            syns2 = wordnet.synsets(word2)
            if syns1 and syns2:
                s = syns1[0].wup_similarity(syns2[0])
                if s > 0.75:
                    if int(items[w1]) not in sims:
                        sims[int(items[w1])] = [int(items[w2])]
                    else:
                        sims[int(items[w1])].append(int(items[w2]))
print(sims)
print()
rule_matrix = {i: [i] for i in range(1, 257)}
updates = {3: [8, 42, 66, 135, 165, 185, 199, 229], 7: [9, 28, 38, 56, 64, 65, 80, 84, 85, 90, 105, 121, 134, 164, 250, 254, 256], 8: [3, 13, 42, 66, 135, 165, 185, 199, 212, 229], 9: [7, 28, 38, 56, 64, 65, 84, 85, 90, 105, 121, 134, 164, 250, 254], 12: [67, 141, 209, 210, 219, 240], 13: [8, 66, 199, 212], 15: [15, 25, 68, 103], 20: [20], 22: [22], 23: [145, 192], 24: [34, 40, 93, 198, 230], 25: [15, 68, 118], 26: [55], 28: [7, 9, 56, 64, 65, 84, 85, 105, 134, 164, 250], 30: [122, 123], 34: [24, 40, 48, 52, 93, 189, 198, 201, 230], 36: [36], 38: [7, 9, 64, 90, 164], 39: [199], 40: [24, 34, 93, 198, 230], 42: [3, 8, 66, 135, 165, 185, 199, 229], 48: [34, 189, 201, 230], 49: [60, 89, 113, 114, 152, 158, 207], 52: [34, 52, 198, 230], 55: [26], 56: [7, 9, 28, 64, 84, 105, 134, 164, 250, 254], 57: [57, 87], 60: [49, 89, 113, 114, 152, 158, 207], 64: [7, 9, 28, 38, 56, 64, 65, 84, 85, 90, 105, 121, 134, 164, 250], 65: [7, 9, 28, 64, 84, 85, 105, 134, 164, 250], 66: [3, 8, 13, 42, 66, 135, 165, 185, 199, 212, 229], 67: [12, 141, 209, 210, 219, 240], 68: [15, 25, 118], 74: [131], 80: [7, 113, 152, 190, 207, 256], 84: [7, 9, 28, 56, 64, 65, 85, 105, 134, 164, 250], 85: [7, 9, 28, 64, 65, 84, 105, 134, 164, 250], 87: [57], 89: [49, 60, 113, 114, 152, 158, 207], 90: [7, 9, 38, 64, 164], 93: [24, 34, 40, 198, 230], 95: [206], 97: [247], 98: [99, 136, 247], 99: [98, 136, 247], 100: [100], 102: [102, 251], 103: [15], 105: [7, 9, 28, 56, 64, 65, 84, 85, 134, 164, 250], 110: [240], 113: [49, 60, 80, 89, 114, 152, 158, 207, 256], 114: [49, 60, 89, 113, 114, 152, 158, 207], 116: [190], 118: [25, 68], 121: [7, 9, 64, 121, 164], 122: [30, 123], 123: [30, 122, 123], 127: [127], 129: [129, 168], 131: [74], 134: [7, 9, 28, 56, 64, 65, 84, 85, 105, 134, 164, 250], 135: [3, 8, 42, 66, 165, 185, 199, 229], 136: [98, 99, 247], 137: [177], 139: [181, 203, 227], 140: [140], 141: [12, 67, 209, 210, 219], 144: [236], 145: [23, 145, 192], 148: [150], 150: [148], 152: [49, 60, 80, 89, 113, 114, 158, 207, 256], 158: [49, 60, 89, 113, 114, 152, 207], 164: [7, 9, 28, 38, 56, 64, 65, 84, 85, 90, 105, 121, 134, 250], 165: [3, 8, 42, 66, 135, 185, 199, 229, 244], 168: [129], 172: [172, 173, 209], 173: [172, 209], 176: [203, 231], 177: [137], 180: [234], 181: [139, 203, 227], 185: [3, 8, 42, 66, 135, 165, 199, 229, 244], 189: [34, 48, 201, 230], 190: [80, 116, 228, 256], 192: [23, 145], 196: [206], 198: [24, 34, 40, 52, 93, 201, 230], 199: [3, 8, 13, 39, 42, 66, 135, 165, 185, 212, 229, 244], 201: [34, 48, 189, 198, 201, 230], 203: [139, 176, 181, 227, 231], 204: [204], 206: [95, 196], 207: [49, 60, 80, 89, 113, 114, 152, 158, 256], 209: [12, 67, 141, 172, 173, 210, 219, 240], 210: [12, 67, 141, 209, 219, 240], 211: [247], 212: [8, 13, 66, 199], 219: [12, 67, 141, 209, 210, 240], 227: [139, 181, 203], 228: [190], 229: [3, 8, 42, 66, 135, 165, 185, 199, 244], 230: [24, 34, 40, 48, 52, 93, 189, 198, 201, 230], 231: [176, 203], 234: [180], 235: [235], 236: [144], 240: [12, 67, 110, 209, 210, 219, 240], 244: [165, 185, 199, 229], 247: [97, 98, 99, 136, 211], 250: [7, 9, 28, 56, 64, 65, 84, 85, 105, 134, 164], 251: [102, 251], 254: [7, 9, 56], 256: [7, 80, 113, 152, 190, 207]}
rule_matrix.update(updates)
print(rule_matrix)
'''
for word1, word2 in product(list1, list2):
    syns1 = wordnet.synsets(word1)
    syns2 = wordnet.synsets(word2)
    for sense1, sense2 in product(syns1, syns2):
        d = wordnet.wup_similarity(sense1, sense2)
        sims.append((d, syns1, syns2))

# Strip numbers and extract only the keys from items dictionary
tags = [key for key in items.keys()]

# Initialize a TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(tags)

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Find the most similar items for each item
most_similar = {}
for idx, item in enumerate(tags):
    # Get the similarity scores for the current item with all others, paired with their indexes
    similarity_scores = list(enumerate(cosine_sim[idx]))
    # Sort the items based on the similarity scores
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    #print(similarity_scores)
    n = sum(i > 0.9 for (a,i) in similarity_scores)
    
    most_similar[item] = [tags[i[0]] for i in similarity_scores[0:n]]  # Top n similar items

# Print the most similar items for each item
for key, value in most_similar.items():
    print(f"{key} is most similar to {value}")
'''