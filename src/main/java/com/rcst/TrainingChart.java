package com.rcst;

import java.awt.Color;
import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.XYSeries;
import org.knowm.xchart.style.markers.SeriesMarkers;

/**
 * Generates training charts using XChart.
 */
public class TrainingChart {

    public static void generate(String csvPath, String outputPath)
        throws IOException {
        Path csvFile = Paths.get(csvPath);
        if (!Files.exists(csvFile)) {
            System.err.println("CSV not found: " + csvPath);
            return;
        }

        List<Integer> epochs = new ArrayList<>();
        List<Double> trainLosses = new ArrayList<>();
        List<Double> valLosses = new ArrayList<>();
        List<Double> learningRates = new ArrayList<>();

        try (BufferedReader br = Files.newBufferedReader(csvFile)) {
            String line = br.readLine(); // skip header
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                epochs.add(Integer.parseInt(parts[0]));
                trainLosses.add(Double.parseDouble(parts[1]));
                valLosses.add(Double.parseDouble(parts[2]));
                learningRates.add(Double.parseDouble(parts[3]));
            }
        }

        // Create loss chart
        XYChart lossChart = new XYChartBuilder()
            .width(800)
            .height(500)
            .title("Training Progress")
            .xAxisTitle("Epoch")
            .yAxisTitle("Loss")
            .build();

        lossChart.getStyler().setLegendVisible(true);
        lossChart
            .getStyler()
            .setDefaultSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Line);
        lossChart.getStyler().setMarkerSize(4);

        XYSeries trainSeries = lossChart.addSeries(
            "Train Loss",
            epochs,
            trainLosses
        );
        trainSeries.setMarker(SeriesMarkers.CIRCLE);
        trainSeries.setLineColor(Color.BLUE);

        XYSeries valSeries = lossChart.addSeries("Val Loss", epochs, valLosses);
        valSeries.setMarker(SeriesMarkers.DIAMOND);
        valSeries.setLineColor(Color.RED);

        // Highlight best validation
        double bestVal = valLosses.stream().min(Double::compare).orElse(0.0);
        int bestIdx = valLosses.indexOf(bestVal);
        lossChart
            .addSeries(
                "Best Val (" + epochs.get(bestIdx) + ")",
                List.of(epochs.get(bestIdx)),
                List.of(bestVal)
            )
            .setMarker(SeriesMarkers.DIAMOND)
            .setLineColor(Color.GREEN);

        // Save
        BitmapEncoder.saveBitmap(
            lossChart,
            outputPath,
            BitmapEncoder.BitmapFormat.PNG
        );
        System.out.println("Chart saved to: " + outputPath + ".png");

        // Print summary
        System.out.println("\nTraining Summary:");
        System.out.printf("  Epochs: %d%n", epochs.size());
        System.out.printf(
            "  Final train loss: %.4f%n",
            trainLosses.get(trainLosses.size() - 1)
        );
        System.out.printf(
            "  Final val loss: %.4f%n",
            valLosses.get(valLosses.size() - 1)
        );
        System.out.printf(
            "  Best val loss: %.4f (epoch %d)%n",
            bestVal,
            epochs.get(bestIdx)
        );
    }

    public static void main(String[] args) throws IOException {
        String csvPath =
            args.length > 0 ? args[0] : "checkpoints/training_log.csv";
        String outputPath =
            args.length > 1 ? args[1] : "checkpoints/training_chart";
        generate(csvPath, outputPath);
    }
}
