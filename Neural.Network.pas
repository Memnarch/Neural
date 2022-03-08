unit Neural.Network;

interface

uses
  Neural,
  Neural.Layers,
  Neural.Scheduler;

type
  TEpochProgression = reference to procedure(ACurrentEpoche: Integer);


  TNeuralNetwork = class
  private
    FOnEpochProgression: TEpochProgression;
    FScheduler: IScheduler;
  protected
    FLayers: TArray<TNeuralLayer>;
    FLongestLayer: Integer;
    FInputCount: Integer;
    FLoss: TLoss;
    procedure CopyValues(var ATarget: TArray<Single>; const ASource: TArray<Single>);
    procedure Backpropagade(const ALearningRate, APartialDerived: Single);
  public
    destructor Destroy; override;
    procedure Build(const AInput: TShape; const ALayers: TArray<TNeuralLayer>; const ALoss: TLoss);
    function FeedForward(const AValues: TArray<Single>): TArray<Single>;
    procedure Train(const AData: TArray<TArray<Single>>; const AExpected: TArray<Single>; AEpochs: Integer = 1000; ALearningRate: Single = 0.01);
    property OnEpochProgression: TEpochProgression read FOnEpochProgression write FOnEpochProgression;
    property Scheduler: IScheduler read FScheduler write FScheduler;
  end;

implementation

{$ExcessPrecision OFF}

uses
  System.Math;

{ TNeuralNetwork }

procedure TNeuralNetwork.Backpropagade(const ALearningRate, APartialDerived: Single);
var
  i: Integer;
  LGradient: TArray<Single>;
begin
  LGradient := [APartialDerived];
  for i := High(FLayers) downto Low(FLayers) do
    LGradient := FLayers[i].Backpropagade(LGradient, ALearningRate);
end;

procedure TNeuralNetwork.Build(const AInput: TShape; const ALayers: TArray<TNeuralLayer>; const ALoss: TLoss);
var
  LPrevOutput: TShape;
  LLayer: TNeuralLayer;
begin
  FLoss := ALoss;
  FInputCount := AInput.Size;
  FLongestLayer := FInputCount;
  FLayers := ALayers;
  LPrevOutput := AInput;
  for LLayer in FLayers do
  begin
    LLayer.Scheduler := FScheduler;
    LLayer.InputShape := LPrevOutput;
    LLayer.Build;
    LPrevOutput := LLayer.OutputShape;
  end;
end;

procedure TNeuralNetwork.CopyValues(var ATarget: TArray<Single>;
  const ASource: TArray<Single>);
var
  i: Integer;
begin
  for i := Low(ASource) to High(ASource) do
    ATarget[i] := ASource[i];
end;

destructor TNeuralNetwork.Destroy;
var
  LLayer: TNeuralLayer;
begin
  for LLayer in FLayers do
    LLayer.Free;
  inherited;
end;

function TNeuralNetwork.FeedForward(const AValues: TArray<Single>): TArray<Single>;
var
  LLayer: TNeuralLayer;
begin
  Result := AValues;
  for LLayer in FLayers do
    Result := LLayer.FeedForward(Result);
end;

procedure TNeuralNetwork.Train(const AData: TArray<TArray<Single>>;
  const AExpected: TArray<Single>; AEpochs: Integer = 1000; ALearningRate: Single = 0.01);
var
  i, k: Integer;
  LResults: TArray<Single>;
  LPartialDerived: Single;
  LUpdateSteps: Integer;
begin
  LUpdateSteps := Max(1, AEpochs div 100);
  for i := 1 to AEpochs do
  begin
    for k := Low(AData) to High(AData) do
    begin
      LResults := FeedForward(AData[k]);

      LPartialDerived := FLoss.Derive([AExpected[k]], [LResults[0]]);
      Backpropagade(ALearningRate, LPartialDerived);
    end;
    if ((i mod LUpdateSteps) = 0) and Assigned(FOnEpochProgression) then
      FOnEpochProgression(i);
  end;
end;

end.
