import { fireEvent, render, screen } from '@testing-library/react';
import App from './App';

test('renders the voltammetry analysis landing UI', () => {
  render(<App />);

  expect(
    screen.getByRole('heading', {
      name: /estimate carboplatin concentration from a single voltammogram/i,
    })
  ).toBeInTheDocument();
  expect(screen.getByText(/drop your csv here or browse/i)).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /run analysis/i })).toBeDisabled();
});

test('lets the user select a mock model architecture', () => {
  render(<App />);

  const pinn = screen.getByRole('radio', { name: /pinn-cv/i });
  const cnn = screen.getByRole('radio', { name: /cnn-transformer/i });

  expect(pinn).toBeChecked();
  expect(cnn).not.toBeChecked();

  fireEvent.click(cnn);

  expect(cnn).toBeChecked();
  expect(pinn).not.toBeChecked();
  expect(
    screen.getByText(/convolutional features are refined by transformer attention/i)
  ).toBeInTheDocument();
});
