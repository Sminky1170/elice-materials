########################## 1. 역대 월드컵 관중수 출력하기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from elice_utils import EliceUtils
elice_utils = EliceUtils()
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
'''
출력 형식을 위한 스켈레톤 코드입니다.
아래 줄 부터 문제에 맞는 코드를 작성해주세요.
'''
world_cups = pd.read_csv("WorldCups.csv")

world_cups = world_cups[['Year', 'Attendance']]

print(world_cups)

########################## 2. 역대 월드컵의 경기당 득점수
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from elice_utils import EliceUtils
elice_utils = EliceUtils()
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
'''
출력 형식을 위한 스켈레톤 코드입니다.
아래 줄 부터 문제에 맞는 코드를 작성해주세요.
'''

world_cups = pd.read_csv("WorldCups.csv")

world_cups = world_cups[["Year", "GoalsScored", "MatchesPlayed"]]

world_cups["GoalsPerMatch"] = world_cups["GoalsScored"] / world_cups["MatchesPlayed"]

print(world_cups)

######################### 3.월드컵 매치 데이터 전처리
import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
'''
출력 형식을 위한 스켈레톤 코드입니다.
아래 줄 부터 문제에 맞는 코드를 작성해주세요.
'''

world_cups_matches = pd.read_csv("./WorldCupMatches.csv")

print("전처리 이전:")
print("Germany FR: {}".format(world_cups_matches.isin(["Germany FR"]).sum().sum()))
print("C�te d'Ivoire: {}".format(world_cups_matches.isin(["C�te d'Ivoire"]).sum().sum()))
print("rn\">Bosnia and Herzegovina: {}".format(world_cups_matches.isin(["rn\">Bosnia and Herzegovina"]).sum().sum()))
print("rn\">Serbia and Montenegro: {}".format(world_cups_matches.isin(["rn\">Serbia and Montenegro"]).sum().sum()))
print("rn\">Trinidad and Tobago: {}".format(world_cups_matches.isin(["rn\">Trinidad and Tobago"]).sum().sum()))
print("rn\">United Arab Emirates".format(world_cups_matches.isin(["rn\">United Arab Emirates"]).sum().sum()))
print("Soviet Union: {}".format(world_cups_matches.isin(["Soviet Union"]).sum().sum()))

# Q1
world_cups_matches = world_cups_matches.replace("Germany FR", "Germany")
world_cups_matches = world_cups_matches.replace("C�te d'Ivoire", "Côte d'Ivoire")
world_cups_matches = world_cups_matches.replace("rn\">Bosnia and Herzegovina", "Bosnia and Herzegovina")
world_cups_matches = world_cups_matches.replace("rn\">Serbia and Montenegro", "Serbia and Montenegro")
world_cups_matches = world_cups_matches.replace("rn\">Trinidad and Tobago", "Trinidad and Tobago")
world_cups_matches = world_cups_matches.replace("rn\">United Arab Emirates", "United Arab Emirates")
world_cups_matches = world_cups_matches.replace("Soviet Union", "Russia")

print("\n전처리 이후:")
print("Germany FR: {}".format(world_cups_matches.isin(["Germany FR"]).sum().sum()))
print("C�te d'Ivoire: {}".format(world_cups_matches.isin(["C�te d'Ivoire"]).sum().sum()))
print("rn\">Bosnia and Herzegovina: {}".format(world_cups_matches.isin(["rn\">Bosnia and Herzegovina"]).sum().sum()))
print("rn\">Serbia and Montenegro: {}".format(world_cups_matches.isin(["rn\">Serbia and Montenegro"]).sum().sum()))
print("rn\">Trinidad and Tobago: {}".format(world_cups_matches.isin(["rn\">Trinidad and Tobago"]).sum().sum()))
print("rn\">United Arab Emirates".format(world_cups_matches.isin(["rn\">United Arab Emirates"]).sum().sum()))
print("Soviet Union: {}".format(world_cups_matches.isin(["Soviet Union"]).sum().sum()))

# Q2
dupli = world_cups_matches.duplicated()
print("\n중복된 값 개수: {}".format(len(dupli[dupli == True])))

world_cups_matches = world_cups_matches.drop_duplicates()


############################3 4. 국가별 득점수 구하기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from elice_utils import EliceUtils
elice_utils = EliceUtils()
import preprocess
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
'''
출력 형식을 위한 스켈레톤 코드입니다.
아래 줄 부터 문제에 맞는 코드를 작성해주세요.
'''

# Q1
world_cups_matches = preprocess.world_cups_matches

# Q2
home = world_cups_matches.groupby(['Home Team Name'])['Home Team Goals'].sum()
away = world_cups_matches.groupby(['Away Team Name'])['Away Team Goals'].sum()

# Q3
goal_per_country = pd.concat([home, away], axis=1, sort=True).fillna(0)

# Q4
goal_per_country["Goals"] = goal_per_country["Home Team Goals"] + goal_per_country["Away Team Goals"]

# Q5
goal_per_country = goal_per_country["Goals"].sort_values(ascending = False)

# Q6
goal_per_country = goal_per_country.astype(int)

print(goal_per_country)


#############################3 5. 2014년 월드컵 다득점 국가순위
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from elice_utils import EliceUtils
elice_utils = EliceUtils()
import preprocess
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
'''
출력 형식을 위한 스켈레톤 코드입니다.
아래 줄 부터 문제에 맞는 코드를 작성해주세요.
'''

world_cups_matches = preprocess.world_cups_matches

# Q1
world_cups_matches = world_cups_matches[world_cups_matches['Year'] == 2014]

# Q2
home_team_goal = world_cups_matches.groupby(['Home Team Name'])['Home Team Goals'].sum()
away_team_goal = world_cups_matches.groupby(['Away Team Name'])['Away Team Goals'].sum()

# Q3
team_goal_2014 = pd.concat([home_team_goal, away_team_goal], axis=1).fillna(0)

# Q4
team_goal_2014['Goals'] = team_goal_2014['Home Team Goals'] + team_goal_2014['Away Team Goals']
team_goal_2014 = team_goal_2014.drop(['Home Team Goals', 'Away Team Goals'], axis=1)

# Q5
team_goal_2014 = team_goal_2014['Goals'].sort_values(ascending=False)

print(team_goal_2014)
