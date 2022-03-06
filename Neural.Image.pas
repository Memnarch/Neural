unit Neural.Image;

interface

uses
  System.Types,
  Graphics;

type
  TNeuralImage = class
  private
    FImage: TBitmap;
  public
    constructor Create();
    destructor Destroy; override;
    procedure LoadFromFile(const AFile: string);
    function ToLumenValues: TArray<Single>;
  end;

implementation

uses
  VCL.Imaging.jpeg;

{ TNeuralImage }

constructor TNeuralImage.Create;
begin
  inherited;
  FImage := TBitmap.Create(64, 64);
  FImage.PixelFormat := pf24bit;
end;

destructor TNeuralImage.Destroy;
begin
  FImage.Free;
  inherited;
end;

procedure TNeuralImage.LoadFromFile(const AFile: string);
var
  LPic: TPicture;
  LSize, LLeft, LTop: Integer;
  LScale: Single;
  LRect: TRect;
begin
  LPic := TPicture.Create();
  try
    LPic.LoadFromFile(AFile);
    FImage.Canvas.Brush.Color := clBlack;
    FImage.Canvas.FillRect(FImage.Canvas.ClipRect);
    if LPic.Width > LPic.Height then
      LSize := LPic.Width
    else
      LSize := LPic.Height;

    LScale := FImage.Width / LSize;
    LRect := TRect.Create(Point(0, 0), Round(LPic.Width*LScale), Round(LPic.Height*LScale));
    LLeft := (FImage.Width - LRect.Width) div 2;
    LTop := (FImage.Height - LRect.Height) div 2;
    LRect.Offset(LLeft, LTop);
    FImage.Canvas.StretchDraw(LRect, LPic.Graphic);
  finally
    LPic.Free;
  end;
end;

type
  TRGB24 = packed record
    R, G, B: Byte;
  end;

  PRGB24 = ^TRGB24;

{$POINTERMATH ON}

function TNeuralImage.ToLumenValues: TArray<Single>;
var
  LPixels: PRGB24;
  i, k: Integer;
  LLumen: Single;
  LCursor: Integer;
begin
  SetLength(Result, FImage.Width*FImage.Height);
  LCursor := 0;
  for i := 0 to Pred(FImage.Height) do
  begin
    LPixels := FImage.ScanLine[i];
    for k := 0 to Pred(FImage.Width) do
    begin
      LLumen := 0.2126 * LPixels[k].R + 0.7152 * LPixels[k].G + 0.0722 * LPixels[k].B;
      Result[LCursor] := LLumen / 255;//normalize
      Inc(LCursor);
    end;
  end;
end;

end.
