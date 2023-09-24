clear
nobs=100;
n=1;
load oil_100.dat;
pp=oil_100;
TT=smooth(pp,0.4,'loess');
for j=1:n
y=trajgen(nobs);
for i=1:nobs
    ppp(i)=pp(i)-TT(i);
    h(i)=abs(y(i)-ppp(i));
 end
hh(j)=mean(h);
end
[mean(hh) std(hh)]
plot(ppp)
hold on
plot(y)