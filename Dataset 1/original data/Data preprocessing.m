clear; clc;

disp('Processing data');

trRaw=dlmread('train_FD001.txt');

trSet.Idx = trRaw(:,1); trSet.Ts = trRaw(:,2); trSet.Raw = trRaw(:,3:end);

trSet.Life = trSet.Ts(diff([trSet.Idx;max(trSet.Idx)+1]) > 0);

tsRaw=dlmread('test_FD001.txt');

tsSet.Idx = tsRaw(:,1); tsSet.Ts = tsRaw(:,2); tsSet.Raw = tsRaw(:,3:end);

tsSet.RUL=dlmread('RUL_FD001.txt');

colnames={'altitudes','Mach','sea-level(temperatures)',...
    'T2','T24','T30','T50','P2','P15','P30','Nf','Nc','epr','Ps30','phi','NRf','NRc',...
    'BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32'};

 

% select sensors

sensor_idx=[5,7,10,11,14,15,16,18,20,23,24]; % these sensors are selected by Liu

trSet.L = trSet.Raw(:,sensor_idx);

tsSet.L = tsSet.Raw(:,sensor_idx);

sensor_trend=[1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1]; % sensors that have a trend

ntr_units=max(trSet.Idx);

nts_units=max(tsSet.Idx);

 

% standardize training data

[trSet.L, mu, sigma]=zscore(trSet.L.*repmat(sensor_trend, size(trSet.L,1), 1));

for i=1:ntr_units

    idx = trSet.Idx == i;

    tmp=trSet.L(idx,:);

    tmp=min(tmp,[],1)-exp(tmp(1,:));

    trSet.L(idx,:)=trSet.L(idx,:)-repmat(tmp,sum(idx),1);

end

trSet.L = log(trSet.L);

trSet.L = trSet.L.*repmat(sensor_trend,size(trSet.L,1),1); % trend changes back

 

% standardize testing data

tsSet.L=(tsSet.L.*repmat(sensor_trend, size(tsSet.L,1),1)...
    -repmat(mu,size(tsSet.L,1),1))./repmat(sigma,size(tsSet.L,1),1);

for i=1:nts_units

    idx = tsSet.Idx == i;

    tmp = tsSet.L(idx,:);

    tmp = min(tmp,[],1)-exp(tmp(1,:));

    tsSet.L(idx,:)=tsSet.L(idx,:)-repmat(tmp,sum(idx),1);

end

tsSet.L = log(tsSet.L);
tsSet.L = tsSet.L.*repmat(sensor_trend, size(tsSet.L,1) ,1);



%%%%%%%%%%%%%%%%%%%%%%%%%%
%save data
TD_data = [trSet.Idx trSet.L];
Test_data = [tsSet.Idx tsSet.L];

xlswrite('TD_data', TD_data)
xlswrite('Test_data', Test_data)




%%%%%%%%%%%%%%%%%%%
for unit=1:100;   
FD_unit=trSet.L(find(trSet.Idx==unit),:);
Time_unit=trSet.Ts(find(trSet.Idx==unit),:); 

%Failure(unit,:)=FD_unit(end,:);
for ip=1:11
    %figure(unit)
subplot (4,3,ip)
plot(Time_unit,FD_unit(:,ip),'.')
axis([-inf inf -inf inf])
end
F=getframe(gcf);
%imwrite(F.cdata,['Unit #',num2str(unit),'.png']) 
end

