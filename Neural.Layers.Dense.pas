unit Neural.Layers.Dense;

interface

uses
  Neural,
  Neural.Layers;

type
  TDenseLayer = class(TNeuralLayer)
  private
    FNeuronCount: Integer;
    FWeights: TArray<TArray<Single>>;
    FBiases: TArray<Single>;
    FLastOutput: TNums;
    FLastFeedForward: TNums;
    FLastBackpropagade: TNums;
    FLastInput: TArray<Single>;
    FBackPropagationBuffer: TArray<TNums>;
    function RunNode(const AInputs, AWeights: TArray<Single>; ABias: Single): Single;
    procedure RunNodeBackwards(const AResult, Gradient, ALearnRate: Single; var AWeights: TArray<Single>; var ABias: Single; var TargetGradients: TNums);
  public
    constructor Create(ACount: Integer; const AActivation: TActivation; const AWeightInitializer: TInitializerFunc = nil); reintroduce;
    procedure Build; override;
    function FeedForward(const Input: TArray<Single>): TArray<Single>; override;
    function Backpropagade(const AGradients: TArray<Single>; const ALearningRate: Single): TArray<Single>; override;
  end;

implementation

uses
  Windows;

{$EXCESSPRECISION OFF}

{ TDenseLayer }

function TDenseLayer.Backpropagade(const AGradients: TArray<Single>;
  const ALearningRate: Single): TArray<Single>;
var
  LResult: TArray<Single>;
  i, k: Integer;
begin
  FLastBackpropagade.FillZero;
  RunScheduled(Length(FWeights),
    procedure(Index: Integer)
    begin
      FBackPropagationBuffer[Index].FillZero;
      RunNodeBackwards(FLastOutput[Index], AGradients[Index], ALearningRate, FWeights[Index], FBiases[Index], FBackPropagationBuffer[Index]);
    end
  );
  Result := FLastBackpropagade.Flat;
  for k := Low(FBackPropagationBuffer) to High(FBackPropagationBuffer) do
  begin
    LResult := FBackPropagationBuffer[k].Flat;
    for i := Low(LResult) to High(LResult) do
      Result[i] := Result[i] + LResult[i];
  end;

end;

procedure TDenseLayer.Build;
var
  i: Integer;
  LInputs: Integer;
begin
  inherited;
  SetLength(FBackPropagationBuffer, FNeuronCount);
  SetLength(FWeights, FNeuronCount);
  SetLength(FBiases, FNeuronCount);
  FOutputShape := TShape.Create(FNeuronCount);
  FLastOutput.Reshape(FOutputShape);
  FLastFeedForward.Reshape(FOutputShape);
  FLastBackpropagade.Reshape(FInputShape);
  LInputs := FInputShape.Size;
  for i := Low(FWeights) to High(FWeights) do
  begin
    FWeights[i] := FWeightInitializer([LInputs]).Flat;
    FBiases[i] := 0;
    FBackPropagationBuffer[i].Reshape([LInputs]);
  end;
end;

constructor TDenseLayer.Create(ACount: Integer; const AActivation: TActivation; const AWeightInitializer: TInitializerFunc = nil);
begin
  inherited Create(AActivation, AWeightInitializer);
  FNeuronCount := ACount;
end;

function TDenseLayer.FeedForward(const Input: TArray<Single>): TArray<Single>;
begin
  FLastInput := Input;
  RunScheduled(Length(FWeights),
    procedure(Index: Integer)
    begin
      FLastOutput[Index] := RunNode(Input, FWeights[Index], FBiases[Index]);
      FLastFeedForward[Index] := FActivation.Run(FLastOutput[Index]);
    end
  );
  Result := FLastFeedForward.Flat;
end;

function TDenseLayer.RunNode(const AInputs, AWeights: TArray<Single>;
  ABias: Single): Single;
var
  i: Integer;
begin
  Result := ABias;
  for i := Low(AWeights) to High(AWeights) do
    Result := Result + AInputs[i] * AWeights[i];
end;

procedure TDenseLayer.RunNodeBackwards(
  const AResult, Gradient, ALearnRate: Single;
  var AWeights: TArray<Single>;
  var ABias: Single;
  var TargetGradients: TNums);
var
  i: Integer;
  LDerived, LGradient: Single;
begin
  LDerived := FActivation.Derive(AResult);
  for i := Low(AWeights) to High(AWeights) do
  begin
    LGradient := FLastInput[i] * LDerived;
    TargetGradients[i] := TargetGradients[i] + AWeights[i] * LDerived * Gradient;
    AWeights[i] := AWeights[i] - (ALearnRate * Gradient * LGradient);
  end;
  ABias := ABias - (ALearnRate * Gradient * LDerived);
end;

end.
