########################### 1.[미션1]가게db구축하기

-- 아래에 문제 설명대로 정보를 추가해 봅시다.
insert into product values(1, 'carrot', 100, '2019-04-10', 1000, 900);
insert into product values(2, 'tea', 1000, '2020-02-10', 1000, 900);
insert into product values(3, 'clock', 100, null, 200000, 180000);

-- product테이블 전체를 조회해 봅시다.
select * from product;




########################### 2.[미션2]가게db 수정하기

-- 아래에 문제 설명대로 수정해 봅시다.
update product set stock = 0 where id = 1;
update product set stock = 50 where id = 3;
update product set selling_price = 800 where id = 2;
delete from product where id = 4;

-- 수정된 product테이블 전체를 조회합니다. 만약 product를 수정하지 않았다면 수정되지 않은 값이 조회됩니다.
select * from product;




########################### 3.[미션3]발언권이 강한주주

-- 아래에 미션을 수행하는 코드를 작성해 봅시다.
select * from shareholder
order by stock desc;



########################### 4.[미션1]가게db 분석하기

-- 아래에 미션을 수행하는 쿼리를 작성해 보세요.
select count(*) from product;
select sum(stock) from product;
select max(selling_price) from product;



########################### 5.[미션2] 평균구하기

-- 지시사항을 만족하는 쿼리를 작성해보세요.
SELECT AVG(math), AVG(eng) FROM test;



########################### 6.[미션1]판매기록 조회하기

-- 판매기록을 product 테이블과 연결해 출력해 봅시다.
-- 이때 product테이블이 중심이 되도록 해 봅시다.
select * from sale right join product on product.id = sale.product_id;




########################### 7.[미션3 인기있는물건

-- 지시사항을 만족하는 쿼리를 작성해보세요.
SELECT name, SUM(amount)
FROM sale
GROUP BY name
HAVING SUM(revenue) >= 50000;

