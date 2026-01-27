function samples=gumbel(m,mu,beta)
u = rand(1,m);  % 生成均匀分布的随机数
samples = mu - beta * log(-log(u));
end 