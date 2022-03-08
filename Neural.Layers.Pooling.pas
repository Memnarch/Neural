unit Neural.Layers.Pooling;

interface

uses
  System.Types,
  Neural,
  Neural.Layers;

type
  TMaxPoolingLayer = class(TNeuralLayer)
  private
    FSize: TSize;
    FLastInput: TArray<Single>;
    FLastOutput: TArray<Single>;
    procedure Sample(X, Y: Integer; const AInput, AOutput: TNums);
    procedure PutBacki(X, Y: Integer; const AInputs, AOutput: TNums; const AHighValue, AGradient: Single);
  public
    constructor Create(const ASize: TSize); reintroduce;
    procedure Build; override;
    function FeedForward(const Input: TArray<Single>): TArray<Single>; override;
    function Backpropagade(const AGradients: TArray<Single>; const ALearningRate: Single): TArray<System.Single>; override;
  end;

implementation

{$EXCESSPRECISION OFF}

{ TMaxPoolingLayer }

function TMaxPoolingLayer.Backpropagade(const AGradients: TArray<Single>;
  const ALearningRate: Single): TArray<System.Single>;
var
  i, k: Integer;
  LInput, LOutput, LLastOutput, LGradients: TNums;
begin
  LInput := TNums.Create(FInputShape, FLastInput);
  LOutput := TNums.Create(FInputShape);
  LLastOutput := TNums.Create(FOutputShape, FLastOutput);
  LGradients := TNums.Create(FOutputShape, AGradients);

  for i := 0 to Pred(FOutputShape.Width) do
  begin
    for k := 0 to Pred(FOutputShape.Height) do
      PutBacki(i * FSize.cx, k * FSize.cy, LInput, LOutput, LLastOutput[i, k], LGradients[i, k]);
  end;
  Result := LOutput.Flat;
end;

procedure TMaxPoolingLayer.Build;
begin
  inherited;
  FOutputShape := TShape.Create(FInputShape.Width div FSize.cx, FInputShape.Height div FSize.cy);
end;

constructor TMaxPoolingLayer.Create(const ASize: TSize);
begin
  inherited Create(TActivation.Identity, nil);
  FSize := ASize;
end;

function TMaxPoolingLayer.FeedForward(
  const Input: TArray<Single>): TArray<Single>;
var
  i, k: Integer;
  LInput, LOutput: TNums;
begin
  LInput := TNums.Create(FInputShape, Input);
  LOutput := TNums.Create(FOutputShape);
  FLastInput := Input;
  for i := 0 to Pred(LOutput.Shape.Width) do
  begin
    for k := 0 to Pred(LOutput.Shape.Height) do
      Sample(i, k, LInput, LOutput);
  end;
  Result := LOutput.Flat;
  FLastOutput := Result;
end;

procedure TMaxPoolingLayer.PutBacki(X, Y: Integer; const AInputs, AOutput: TNums; const AHighValue, AGradient: Single);
var
  i, k: Integer;
begin
  for i := X to X + Pred(FSize.cx) do
  begin
    for k := Y to Y +Pred(FSize.cy) do
    begin
      if AInputs[i, k] <> AHighValue then
        AOutput[i, k] := 0
      else
        AOutput[i, k] := AGradient;
    end;
  end;
end;

procedure TMaxPoolingLayer.Sample(X, Y: Integer; const AInput, AOutput: TNums);
var
  i, k: Integer;
  LVal, LCurrent: Single;
begin
  LVal := 0;
  for i := 0 to Pred(FSize.cx) do
  begin
    for k := 0 to Pred(FSize.cy) do
    begin
      LCurrent := AInput[X*FSize.cx + i, Y*FSize.cy + k];
      if LCurrent > LVal then
        LVal := LCurrent;
    end;
  end;
  AOutput[X, Y] := LVal;
end;

end.
