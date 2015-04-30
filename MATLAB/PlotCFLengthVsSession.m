function PlotCFLengthVsSession(D)

if ~exist('D', 'var') || isempty(D)
    D1 = load('C:\E\Dropbox\Lab\MatlabData\TwoPhoton\Feng_Bcd_GFP_TwoPhoton_2012.mat');
    D = D1.BCDEm.M;
end

NumSessions = length(D);
NumCFsC = NaN(NumSessions, 1);
MeanCF_C = NaN(NumSessions, 1);
SD_CF_C = NaN(NumSessions, 1);
%SEM_CF_C = NaN(NumSessions, 1);
for lSession = 1:NumSessions
    if strcmp(D(lSession).Em(1).ID.FlyLine0, 'BCD20A')
        NumEmbryos = length(D(lSession).Em);
        CFflagLR = false(1, NumEmbryos);
        for lEmbryo = 1:NumEmbryos
            if D(lSession).Em(lEmbryo).Prop.CFflag
                CFflagLR(lEmbryo) = true;
            end
        end
        ValidEmLR = CFflagLR & ~isnan([D(lSession).Em.CF]);
        
        CFThisSessionR = [D(lSession).Em.CF];
        ValidCF_R = CFThisSessionR(ValidEmLR);
        
        NumCFsC(lSession) = length(ValidCF_R);
        MeanCF_C(lSession) = mean(ValidCF_R);
        SD_CF_C(lSession) = std(ValidCF_R);
    end
end

MarkerSizeFactor = 10;
figure('WindowStyle', 'docked');
hold all;
scatter(MeanCF_C, SD_CF_C, MarkerSizeFactor*NumCFsC, 'c', 'filled');
for lSession = 1:NumSessions
    text(MeanCF_C(lSession), SD_CF_C(lSession), sprintf('%d', lSession));
end
xlabel('Mean CF position (% egg length; anterior pole = 0%)');
ylabel('Standard deviation of CF position (% egg length)');
title({'Cephalic furrow (CF) position for each imaging session', '(number of embryos proportional to marker size)'});

end