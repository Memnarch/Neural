unit Neural;

interface

type
  TShape = array of Integer;

  TShapeHelper = record helper for TShape
  private
    function GetSize: Integer; inline;
    function GetDepth: Integer; inline;
    function GetHeight: Integer; inline;
    function GetWidth: Integer; inline;
  public
    property Width: Integer read GetWidth;
    property Height: Integer read GetHeight;
    property Depth: Integer read GetDepth;
    property Size: Integer read GetSize;
  end;

  TNums = packed record
  private
    FData: TArray<Single>;
    FShape: TShape;
    function GetValue(X: Integer): Single; inline;
    function GetValue2(X, Y: Integer): Single; inline;
    function GetValue3(X, Y, Z: Integer): Single; inline;
    function GetFlat: TArray<Single>; inline;
    procedure SetValue(X: Integer; const Value: Single); inline;
    procedure SetValue2(X, Y: Integer; const Value: Single); inline;
    procedure SetValue3(X, Y, Z: Integer; const Value: Single); inline;
  public
    class function Create(const AShape: TShape; const AValues: TArray<Single> = nil): TNums; static;
    procedure Reshape(const AShape: TShape);
    function Ref(X: Integer): PSingle; overload;
    function Ref(X, Y: Integer): PSingle; overload;
    function Ref(X, Y, Z: Integer): PSingle; overload;
    class operator Multiply(const ALeft: TNums; const Value: Single): TNums; inline;
    class operator Divide(const ALeft: TNums; const Value: Single): TNums; inline;
    class operator Add(const ALeft: TNums; const Value: Single): TNums; inline;
    class operator Subtract(const ALeft: TNums; const Value: Single): TNums; inline;
    property Value[X: Integer]: Single read GetValue write SetValue; default;
    property Value[X, Y: Integer]: Single read GetValue2 write SetValue2; default;
    property Value[X, Y, Z: Integer]: Single read GetValue3 write SetValue3; default;
    property Flat: TArray<Single> read GetFlat;
    property Shape: TShape read FShape;
  end;

  TActivationFunc = function(const Value: Single): Single;

  TActivation = packed record
    Run: TActivationFunc;
    Derive: TActivationFunc;
    class function Sigmoid: TActivation; static;
    class function Relu: TActivation; static;
    class function Identity: TActivation; static;
  end;

  TLossFunc = function(const AExpected, ACalculated: TArray<Single>): Single;

  TLoss = packed record
    Run: TLossFunc;
    Derive: TLossFunc;
    class function MSE: TLoss; static;
    class function BinaryCrossEntropy: TLoss; static;
  end;

  TInitializerFunc = reference to function(const AShape: TShape): TNums;

  TInitializers = record
    class function HeUniform: TInitializerFunc; static;
    class function Random: TInitializerFunc; static;
    class function Fixed(const Values: TArray<Single>): TInitializerFunc; static;
    class function Default: TInitializerFunc; static;
    class function Ones: TInitializerFunc; static;
  end;

//make compiler happy
function CheckBounds(const AShape: TShape; X: Integer): Boolean; overload;
function CheckBounds(const AShape: TShape; X, Y: Integer): Boolean; overload;
function CheckBounds(const AShape: TShape; X, Y, Z: Integer): Boolean; overload;

implementation

{$ExcessPrecision OFF}

uses
  Math,
  Neural.Math;

{ TShape }

function TShapeHelper.GetDepth: Integer;
begin
  Assert(Length(Self) > 2);
  Result := Self[2];
end;

function TShapeHelper.GetHeight: Integer;
begin
  Assert(Length(Self) > 1);
  Result := Self[1];
end;

function TShapeHelper.GetSize: Integer;
var
  LVal: Integer;
begin
  Result := 1;
  for LVal in Self do
    Result := Result * LVal;
end;


{ TActivation }

function IdentityVal(const Value: Single): Single;
begin
  Result := Value;
end;

class function TActivation.Identity: TActivation;
begin
  Result.Run := IdentityVal;
  Result.Derive := IdentityVal;
end;

class function TActivation.Relu: TActivation;
begin
  Result.Run := Neural.Math.Relu;
  Result.Derive := Neural.Math.DerivedRelu;
end;

class function TActivation.Sigmoid: TActivation;
begin
  Result.Run := Neural.Math.Sigmoid;
  Result.Derive := Neural.Math.DerivedSigmoid;
end;

{ TLoss }

function BCE(const AExpected, ACalculated: TArray<Single>): Single;
var
  i, LLabel: Integer;
  LCalc: Single;
begin
  Result := 0;
  for i := Low(AExpected) to High(AExpected) do
  begin
    LLabel := Round(AExpected[i]);
    LCalc := ACalculated[i];
    if LLabel = 0 then
      Result := Result + Ln(LCalc)
    else if LCalc <> 1 then
      Result := Result + Ln(1-LCalc);
  end;
  Result := -(Result / Length(AExpected));
end;

function DerivedBCE(const AExpected, ACalculated: TArray<Single>): Single;
var
  i: Integer;
  LCalc: Single;
  LLabel: Integer;
begin
  Result := 0;
  for i := Low(AExpected) to High(AExpected) do
  begin
    LCalc := ACalculated[i];
    LLabel := Round(AExpected[i]);
    if LLabel = 1 then
      Result := Result + (1 / LCalc)
    else if LCalc <> 1 then
      Result := Result + (1 / (1 - LCalc));
  end;
  Result := -(Result / Length(AExpected));
end;

class function TLoss.BinaryCrossEntropy: TLoss;
begin
  Result.Run := BCE;
  Result.Derive := DerivedBCE;
end;

class function TLoss.MSE: TLoss;
begin
  Result.Run := MSELoss;
  Result.Derive := DerivedMSELoss;
end;

{ TNums }

function CheckBounds(const AShape: TShape; X: Integer): Boolean; overload;
begin
  Result := (Length(AShape) > 0) and (X < AShape[0]);
end;

function CheckBounds(const AShape: TShape; X, Y: Integer): Boolean; overload;
begin
  Result := (Length(AShape) > 1) and (X < AShape[0]) and (Y < AShape[1]);
end;

function CheckBounds(const AShape: TShape; X, Y, Z: Integer): Boolean; overload;
begin
  Result := (Length(AShape) > 2) and (X < AShape[0]) and (Y < AShape[1]) and (Z < AShape[2]);
end;


class operator TNums.Add(const ALeft: TNums; const Value: Single): TNums;
var
  i: Integer;
begin
  Result := ALeft;
  for i := Low(Result.FData) to High(Result.FData) do
    Result.FData[i] := Result.FData[i] + Value;
end;

class function TNums.Create(const AShape: TShape;
  const AValues: TArray<Single>): TNums;
begin
  Result.FData := AValues;
  Result.Reshape(AShape);
end;

function TNums.Ref(X: Integer): PSingle;
begin
  Assert(CheckBounds(FShape, X));
  Result := @FData[X];
end;

function TNums.Ref(X, Y: Integer): PSingle;
begin
  Assert(CheckBounds(FShape, X, Y));
  Result := @FData[FShape[0]*Y + X];
end;

function TNums.Ref(X, Y, Z: Integer): PSingle;
begin
  Assert(CheckBounds(FShape, X, Y, Z));
  Result := @FData[(FShape[0] * FShape[1] * Z)  + FShape[0]*Y + X];
end;

class operator TNums.Divide(const ALeft: TNums; const Value: Single): TNums;
var
  i: Integer;
begin
  Result := ALeft;
  for i := Low(Result.FData) to High(Result.FData) do
    Result.FData[i] := Result.FData[i] / Value;
end;

function TNums.GetFlat: TArray<Single>;
var
  LSize: Integer;
begin
  Result := FData;
  LSize := FShape.Size;
  if LSize < Length(Result) then
    SetLength(Result, LSize);
end;

function TNums.GetValue(X: Integer): Single;
begin
  Assert(CheckBounds(FShape, X));
  Result := FData[X];
end;

function TNums.GetValue2(X, Y: Integer): Single;
begin
  Assert(CheckBounds(FShape, X, Y));
  Result := FData[FShape[0] * Y + X];
end;

function TNums.GetValue3(X, Y, Z: Integer): Single;
begin
  Assert(CheckBounds(FShape, X, Y, Z));
  Result := FData[(FShape[0] * FShape[1] * Z)  + FShape[0]*Y + X];
end;

class operator TNums.Multiply(const ALeft: TNums; const Value: Single): TNums;
var
  i: Integer;
begin
  Result := ALeft;
  for i := Low(Result.FData) to High(Result.FData) do
    Result.FData[i] := Result.FData[i] * Value;
end;

procedure TNums.Reshape(const AShape: TShape);
var
  LSize: Integer;
begin
  FShape := AShape;
  LSize := FShape.Size;
  if LSize > Length(FData) then
    SetLength(FData, LSize);
end;

procedure TNums.SetValue(X: Integer; const Value: Single);
begin
  Assert(CheckBounds(FShape, X));
  FData[X] := Value;
end;

procedure TNums.SetValue2(X, Y: Integer; const Value: Single);
begin
  Assert(CheckBounds(FShape, X, Y));
  FData[FShape[0] * Y + X] := Value;
end;

procedure TNums.SetValue3(X, Y, Z: Integer; const Value: Single);
begin
  Assert(CheckBounds(FShape, X, Y, Z));
  FData[(FShape[0] * FShape[1] * Z) + (FShape[0] * Y) + X] := Value;
end;

class operator TNums.Subtract(const ALeft: TNums; const Value: Single): TNums;
var
  i: Integer;
begin
  Result := ALeft;
  for i := Low(Result.FData) to High(Result.FData) do
    Result.FData[i] := Result.FData[i] - Value;
end;

function TShapeHelper.GetWidth: Integer;
begin
  Assert(Length(Self) > 0);
  Result := Self[0];
end;

{ TInitializers }

class function TInitializers.Default: TInitializerFunc;
begin
  Result := TInitializers.HeUniform();
end;

class function TInitializers.Fixed(
  const Values: TArray<Single>): TInitializerFunc;
begin
  Result := function(const AShape: TShape): TNums
            begin
              Result := TNums.Create(AShape, Copy(Values));
            end;
end;

class function TInitializers.HeUniform: TInitializerFunc;
begin
  Result := function(const AShape: TShape): TNums
            var
              i, LCount: Integer;
              LLimit, LScale: Single;
            const
              CMax = 1000;
            begin
              Result := TNums.Create(AShape);
              LCount := Length(Result.Flat);
              LLimit := Sqr(6 / LCount);
              for i := Low(Result.Flat) to High(Result.Flat) do
              begin
                LScale := System.Random(CMax) / CMax;
                Result.Flat[i] := (LScale * LLimit * 2) - LLimit;
              end;
            end;
end;

class function TInitializers.Ones: TInitializerFunc;
begin
  Result := function(const AShape: TShape): TNums
            var
              i: Integer;
            begin
              Result := TNums.Create(AShape);
              for i := Low(Result.Flat) to High(Result.Flat) do
                Result.Flat[i] := 1;
            end;
end;

class function TInitializers.Random: TInitializerFunc;
begin
  Result := function(const AShape: TShape): TNums
            var
              i, LCount: Integer;
              LBase: Single;
            const
              CMax = 1000;
            begin
              Result := TNums.Create(AShape);
              LCount := Length(Result.Flat);
              LBase := 1 / LCount;
              for i := Low(Result.Flat) to High(Result.Flat) do
                Result.Flat[i] := (System.Random(CMax) / (CMax div 2) - 1) * LBase;
            end;
end;

end.
