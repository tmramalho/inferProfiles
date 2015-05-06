function PlotCFPosErrorAndScaling(D)

if ~exist('D', 'var') || isempty(D)
    D1 = load('C:\E\Dropbox\Lab\MatlabData\TwoPhoton\Feng_Bcd_GFP_TwoPhoton_2012.mat');
    D = D1.BCDEm.M;
end

NumSessions = length(D);
NumCFsC = NaN(NumSessions, 1);
MeanCF_C = NaN(NumSessions, 1);
SD_CF_C = NaN(NumSessions, 1);
%SEM_CF_C = NaN(NumSessions, 1);
AllCF_R = [];
AllEL_R = [];
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
        AllCF_R = [AllCF_R, CFThisSessionR(ValidEmLR)];
        ELThisSessionR = NaN(1, NumEmbryos);
        for lEmbryo = 1:NumEmbryos
           ELThisSessionR(lEmbryo) = D(lSession).Em(lEmbryo).Prop.Egglength; 
        end
        AllEL_R = [AllEL_R, ELThisSessionR(ValidEmLR)];
    end
end

[CFHistCounts, CFHistCenters] = hist(AllCF_R, 20);
fprintf('Std of CF: %02.1f\n', std(AllCF_R));



%% Figure properties
BasePathS = 'C:\E\Dropbox\Lab\05_Scaling\Plots\inferProfiles\PlotCFPosErrorAndScaling';
if ~exist(BasePathS, 'dir')
    mkdir(BasePathS);
end
DateS = datestr(datenum(date, 'dd-mmm-yyyy'), 'yyyy-mm-dd');
SavePathS = [BasePathS, filesep, DateS];
if ~exist(SavePathS, 'dir')
    mkdir(SavePathS);
end

FontSize = 14;
%LegendFontSize = 12;
LetterFontSize = 21;
AxesHeight = 6; % In centimeters
AxesWidth = 6; % In centimeters
Margin = 2.5; % In centimeters
LetterPositionR = [-0.23, 1.07];

FigWidth = 3*Margin + 2*AxesWidth;
FigHeight = 2*Margin + AxesHeight;
Fig = figure('PaperUnits', 'centimeters',...
    'PaperPosition', [0 0 FigWidth FigHeight],...
    'PaperPositionMode', 'manual',...
    'PaperSize', [FigWidth FigHeight],...
    'Units', 'centimeters',...
    'Position', [0 0 FigWidth FigHeight],...
    'WindowStyle', 'docked');



%% Plotting

axes('Units', 'centimeters', 'FontSize', FontSize,...
    'Position', [Margin, Margin, AxesWidth, AxesHeight]);
hold all;
%text('Units', 'normalized', 'Position', LetterPositionR, 'String', 'A', 'FontSize', LetterFontSize);

bar(CFHistCenters, CFHistCounts);
xlabel('CF position (%EL)');
xlim([30, 40]);
set(gca, 'XTick', 30:2:40);
ylabel('Counts');
ylim([0, 52]);
axis square;
box on;

axes('Units', 'centimeters', 'FontSize', FontSize,...
    'Position', [2*Margin+AxesWidth, Margin, AxesWidth, AxesHeight]);
hold all;
%text('Units', 'normalized', 'Position', LetterPositionR, 'String', 'B', 'FontSize', LetterFontSize);

XLimR = [433, 550];
AbsCF_R = AllCF_R .* AllEL_R / 100;

scatter(AllEL_R, AllCF_R .* AllEL_R / 100, 75, 'k', '.');

PolyFitR = polyfit(AllEL_R, AbsCF_R, 1);
plot(XLimR, polyval(PolyFitR, XLimR), 'Color', [0.5, 0.5, 0.5], 'LineWidth', 1.5);

BinR = 450:10:540;
MeanInBinR = NaN(size(BinR));
for lBin = 2:length(MeanInBinR)-1
    MinEL = (BinR(lBin-1) + BinR(lBin))/2;
    MaxEL = (BinR(lBin) + BinR(lBin+1))/2;
    InBinLR = MinEL <= AllEL_R & AllEL_R < MaxEL;
    MeanInBinR(lBin) = mean(AbsCF_R(InBinLR));
end
scatter(BinR, MeanInBinR, 100, 'r', 'x', 'LineWidth', 1.5);

xlabel('Egg length ({\mu}m)');
xlim(XLimR);
ylabel('CF position ({\mu}m)');
axis square;
box on;



%% Saving
FigNameS = sprintf('PlotCFPosErrorAndScaling');
saveas(gcf, [SavePathS, filesep, FigNameS, '.fig']);
print('-dpdf', '-r600', [SavePathS, filesep, FigNameS, '.pdf']);
print('-dpng', '-r600', [SavePathS, filesep, FigNameS, '.png']);

end