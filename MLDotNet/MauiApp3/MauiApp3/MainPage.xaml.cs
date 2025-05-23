﻿using SkiaSharp.Views.Maui;
using SkiaSharp;
using SkiaSharp.Views.Maui.Controls;

namespace MauiApp3;

public partial class MainPage : ContentPage
{
    private SKPoint? touchLocation;

    public MainPage()
    {
        InitializeComponent();
    }

    private void OnTouch(object sender, SKTouchEventArgs e)
    {
        if (e.InContact)
            touchLocation = e.Location;
        else
            touchLocation = null;

        skiaView.InvalidateSurface();

        e.Handled = true;
    }

    private void OnPaintSurface(object sender, SKPaintSurfaceEventArgs e)
    {
        // the the canvas and properties
        var canvas = e.Surface.Canvas;

        // make sure the canvas is blank
        canvas.Clear(SKColors.White);

        // decide what the text looks like
        using var paint = new SKPaint
        {
            Color = SKColors.Black,
            IsAntialias = true,
            Style = SKPaintStyle.Fill,
            TextAlign = SKTextAlign.Center,
            TextSize = 24
        };

        // adjust the location based on the pointer
        var coord = (touchLocation is SKPoint loc)
            ? new SKPoint(loc.X, loc.Y)
            : new SKPoint(e.Info.Width / 2, (e.Info.Height + paint.TextSize) / 2);

        // draw some text
        canvas.DrawText("SkiaSharp", coord, paint);
    }
}

