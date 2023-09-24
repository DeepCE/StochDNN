clear
%nobs=2200-365;
nobs=1000;
for j=1:100
    y=trajgen(nobs);
    for i=1:nobs-1
        h(i)=y(i+1)-y(i);
    end
    m(j)=mean(h);
    s(j)=std(h);
    sk(j)=skewness(h);
    k(j)=kurtosis(h);
    mi(j)=min(h);
    ma(j)=max(h);
end
[mean(m),mean(s),mean(sk),mean(k)]
[std(m),std(s),std(sk),std(k)]
%plot(y)