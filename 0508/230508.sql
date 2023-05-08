
#################### 1. 헬스장db 만들기
-- member 테이블을 정의하세요.
CREATE TABLE member(
    id          INT         PRIMARY KEY,
    name        VARCHAR(10),
    start_date  DATE,
    end_date    DATE
);


-- member 테이블의 구조를 출력하세요.
DESC member;







##################### 2. 헬스장db 회원추가
-- member 테이블에 데이터를 넣으세요.
INSERT INTO member
VALUES (1001, '김도윤', '2022-08-15', '2023-08-15');
INSERT INTO member
VALUES (1002, '이지아', '2022-09-01', '2022-12-31');

-- member 테이블에 넣은 데이터를 출력하세요.
SELECT * FROM member;



##################### 3. 서점db 관리하기


-- bookstore 테이블에 제약 조건을 추가하세요.
ALTER TABLE bookstore MODIFY COLUMN name VARCHAR(50) NOT NULL;

ALTER TABLE bookstore
ALTER author SET DEFAULT '홍길동';

ALTER TABLE bookstore
ADD CONSTRAINT bookstore_price_check CHECK (price >= 0);





##################### 4. 서점db 인덱스 추가
-- bookstore 테이블에 인덱스를 생성하세요.
CREATE INDEX bookstore_name_index ON bookstore (name);

-- 인덱스가 올바르게 설정되었는지 확인하는 명령어입니다. 해당 코드는 수정하지 마세요.
SHOW INDEX FROM bookstore;



##################### 5. [미션1] 신체검사표
-- 아래에 미션을 수행하는 코드를 작성해 봅시다.
select name from student;
select * from student where gender = 'M';
select height from student where height <= 170;
select weight from student where weight >= 50;



##################### 6. [미션2] 체지방율 검사
-- 아래에 미션을 수행하는 코드를 작성해 봅시다.
select *, weight / ((height/100)*(height/100))
from student;

select *, weight / ((height/100)*(height/100))
from student
where weight / ((height/100)*(height/100)) <= 18.5 or weight / ((height/100)*(height/100)) >=25.0;


##################### 7. [미션3] 주주총회
-- 아래에 미션을 수행하는 코드를 작성해 봅시다.
select * from shareholder;
select * from shareholder where stock >= 100000;
select stock from shareholder where name in ("Alexis", "Craig", "Fred");
select name,stock from shareholder where agree = 0 and stock >= 100000;
select name,stock from shareholder where agree = 1 and stock >= 100000;
select * from shareholder where stock <= 100000 or stock >= 200000;