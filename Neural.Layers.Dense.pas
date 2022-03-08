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
    FLastOutput: TArray<Single>;
    FLastInput: TArray<Single>;
    FBackPropagationBuffer: TArray<TArray<Single>>;
    function RunNode(const AInputs, AWeights: TArray<Single>; ABias: Single): Single;
    procedure RunNodeBackwards(const AResult, Gradient, ALearnRate: Single; var AWeights: TArray<Single>; var ABias: Single; var TargetGradients: TArray<Single>);
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
  i: Integer;
begin
  SetLength(Result, Length(FLastInput));
  ZeroMemory(@Result[0], Length(Result) * SizeOf(Result[0]));
  RunScheduled(Length(FWeights),
    procedure(Index: Integer)
    begin
      ZeroMemory(@FBackPropagationBuffer[Index, 0], Length(FBackPropagationBuffer[Index]) * SizeOf(Single));
      RunNodeBackwards(FLastOutput[Index], AGradients[Index], ALearningRate, FWeights[Index], FBiases[Index], FBackPropagationBuffer[Index]);
    end
  );
  for LResult in FBackPropagationBuffer do
  begin
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
  LInputs := FInputShape.Size;
  for i := Low(FWeights) to High(FWeights) do
  begin
    FWeights[i] := FWeightInitializer([LInputs]).Flat;
    FBiases[i] := 0;
    SetLength(FBackPropagationBuffer[i], LInputs);
  end;
end;

constructor TDenseLayer.Create(ACount: Integer; const AActivation: TActivation; const AWeightInitializer: TInitializerFunc = nil);
begin
  inherited Create(AActivation, AWeightInitializer);
  FNeuronCount := ACount;
end;

function TDenseLayer.FeedForward(const Input: TArray<Single>): TArray<Single>;
var
  LResult: TArray<Single>;
begin
  FLastInput := Input;
  SetLength(LResult, FOutputShape.Size);
  SetLength(FLastOutput, FOutputShape.Size);
  RunScheduled(Length(FWeights),
    procedure(Index: Integer)
    begin
      FLastOutput[Index] := RunNode(Input, FWeights[Index], FBiases[Index]);
      LResult[Index] := FActivation.Run(FLastOutput[Index]);
    end
  );
  Result := LResult;
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
  var TargetGradients: TArray<Single>);
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
