####################################### 1. shape, head()
# 아래 코드는 문제 해결을 위해 기본적으로 제공되는 코드입니다. 수정하지 마세요!
import pandas as pd
import numpy as np

# 지시사항 1번을 참고하여 코드를 작성하세요.
test_data = pd.read_csv('testfile.csv')


# 지시사항 2번을 참고하여 코드를 작성하세요.
print(test_data.shape)


# 지시사항 3번을 참고하여 코드를 작성하세요.
print(test_data.head())

####################################### 2. info()


# 아래 코드는 문제 해결을 위해 기본적으로 제공되는 코드입니다. 수정하지 마세요!
import pandas as pd
import numpy as np

# 지시사항 1번을 참고하여 코드를 작성하세요.
test_data = pd.read_csv('testfile.csv')


# 지시사항 2번을 참고하여 코드를 작성하세요.
print(test_data.info())



####################################### 3. columns

# 아래 코드는 문제 해결을 위해 기본적으로 제공되는 코드입니다. 수정하지 마세요!
import pandas as pd
import numpy as np

# 지시사항 1번을 참고하여 코드를 작성하세요.
test_data = pd.read_csv('testfile.csv')


# 지시사항 2번을 참고하여 코드를 작성하세요.
print(test_data.columns)





###################################### 4. 특정열 선택

# 아래 코드는 문제 해결을 위해 기본적으로 제공되는 코드입니다. 수정하지 마세요!
import numpy as np
import pandas as pd
test_data = pd.read_csv('testfile.csv')

# 지시사항 1번을 참고하여 코드를 작성하세요.
print(test_data.head())


# 지시사항 2번을 참고하여 코드를 작성하세요.
print(test_data[['math']])



###################################### 5. astype()

# 아래 코드는 문제 해결을 위해 기본적으로 제공되는 코드입니다. 수정하지 마세요!
import numpy as np
import pandas as pd
test_data = pd.read_csv('testfile.csv')

# 지시사항 1번을 참고하여 코드를 작성하세요.
print(test_data.info())


# 지시사항 2번을 참고하여 코드를 작성하세요.
test_data = test_data.astype({'class':'category'})


# 지시사항 3번을 참고하여 코드를 작성하세요.
print(test_data.info())



###################################### 6. rename()

# 아래 코드는 문제 해결을 위해 기본적으로 제공되는 코드입니다. 수정하지 마세요!
import numpy as np
import pandas as pd
test_data = pd.read_csv('testfile.csv')

# 지시사항 1번을 참고하여 코드를 작성하세요.
print(test_data.columns)


# 지시사항 2번을 참고하여 코드를 작성하세요.
test_data.rename(columns = {'name' : '이름', 'class' : '학급명', 'math' : '수학', 'english' : '영어', 'korean' : '국어'}, inplace = True)
print(test_data.columns)



###################################### 7. index, isin()

# 아래 코드는 문제 해결을 위해 기본적으로 제공되는 코드입니다. 수정하지 마세요!
import numpy as np
import pandas as pd
test_data = pd.read_csv('testfile.csv')

# 지시사항 1번을 참고하여 코드를 작성하세요.
print(test_data.index)



# 지시사항 2번을 참고하여 코드를 작성하세요.
print(test_data.isin([99,100]))


###################################### 8. value_counts()

# 아래 코드는 문제 해결을 위해 기본적으로 제공되는 코드입니다. 수정하지 마세요!
import numpy as np
import pandas as pd
test_data = pd.read_csv('testfile.csv')

# 지시사항 1번을 참고하여 코드를 작성하세요.
print(test_data['class'].value_counts())




###################################### 9.unique()

# 아래 코드는 문제 해결을 위해 기본적으로 제공되는 코드입니다. 수정하지 마세요!
import numpy as np
import pandas as pd
test_data = pd.read_csv('testfile.csv')

# 지시사항 1번을 참고하여 코드를 작성하세요.

print(test_data['class'].unique())


###################################### 10.barh()

# 아래 코드는 문제 해결을 위해 기본적으로 제공되는 코드입니다. 수정하지 마세요!
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from elice_utils import EliceUtils
elice_utils = EliceUtils()
test_data = pd.read_csv('testfile.csv')

# 지시사항 1번을 참고하여 코드를 작성하세요.
fig, axes = plt.subplots(figsize = (10,7))
axes.barh(test_data['class'], test_data['math'], height = 0.7)
axes.set_title("엘리스 학교 학급 당 평균 수학점수", size=20)
axes.set_xlabel("평균 점수", size=10)
axes.set_ylabel("학급 별", size=10)
plt.margins(y=0.3)

# 아래 코드는 채점을 위한 코드입니다. 수정하지 마세요!
plt.savefig("img.svg", format="svg")
elice_utils.send_image("img.svg")

###################################### 11. bar()

# 아래 코드는 문제 해결을 위해 기본적으로 제공되는 코드입니다. 수정하지 마세요!
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from elice_utils import EliceUtils

elice_utils = EliceUtils()

test_data = pd.read_csv('testfile.csv')

# 지시사항 1번을 참고하여 코드를 작성하세요.
counts = test_data['class'].value_counts().sort_index()
print(counts)

# 지시사항 2번을 참고하여 코드를 작성하세요.
fig, axes = plt.subplots(figsize=(10,10))
axes.bar(counts.index, counts.values)
axes.set_title("학급 별 학생 수", size=30)
axes.set_xlabel("학급 구분", size=20)
axes.set_ylabel("학생 수", size=20)



# 아래 코드는 채점을 위한 코드입니다. 수정하지 마세요!
plt.savefig("img.svg", format="svg")
elice_utils.send_image("img.svg")