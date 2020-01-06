import torch
from torch.autograd import Variable
import time
import os
import sys

from utils import AverageMeter, calculate_RMSE

def train_epoch(epoch, logger, batch_logger, dataloader, optimizer, criterion, model, opt, device):
  print('train at epoch {}'.format(epoch))
  model.train()   
  
  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  errors = AverageMeter()
  
  end_time = time.time()
  
  for i_batch, sample_batched in enumerate(dataloader):
      data_time.update(time.time() - end_time)
      
      inputs = sample_batched['inputs']
      js_input = sample_batched['js']
      inputs, js_input = inputs.to(device), js_input.to(device)
      
      #js_input = js_input.cuda(async=True)
      
      inputs = Variable(inputs)
      js_input = Variable(js_input)
      outputs = model(inputs)
      
      loss = criterion(outputs, js_input)
      err = calculate_RMSE(outputs, js_input)
      
      losses.update(loss.data, inputs.size(0))
      errors.update(err, inputs.size(0))

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      
      
      batch_time.update(time.time() - end_time)
      end_time = time.time()

      batch_logger.log({
          'epoch': epoch,
          'batch': i_batch + 1,
          'iter': (epoch - 1) * len(dataloader) + (i_batch + 1),
          'loss': losses.val,
          'err': errors.val,
          'lr': optimizer.param_groups[0]['lr']
      })

      print('Epoch/batch/num_batch: [{0}]/[{1}/{2}]\t'
            'Batch train time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data read time {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Error {err.val:.3f} ({err.avg:.3f})'.format(
                epoch,
                i_batch + 1,
                len(dataloader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                err=errors))
  
  logger.log({
      'epoch': epoch,
      'loss': losses.avg,
      'err': errors.avg,
      'lr': optimizer.param_groups[0]['lr']
  })
  
  if epoch % opt.checkpoint == 0:
      save_file_path = os.path.join(opt.result_path,
                                    'save_{}.pth'.format(epoch))
      states = {
          'epoch': epoch + 1,
          'arch': opt.arch,
          'state_dict': model.state_dict(),
          'optimizer': optimizer.state_dict(),
      }
      torch.save(states, save_file_path)

def val_epoch(epoch, logger, dataloader, optimizer, criterion, model, device):
    with torch.no_grad():
      print('validation at epoch {}'.format(epoch))
      model.eval()  
      
      batch_time = AverageMeter()
      data_time = AverageMeter()
      losses = AverageMeter()
      errors = AverageMeter()
      
      end_time = time.time()
      
      for i_batch, sample_batched in enumerate(dataloader):
          #print(len(dataloader))
          data_time.update(time.time() - end_time)
          
          inputs = sample_batched['inputs']
          js_input = sample_batched['js']
          inputs, js_input = inputs.to(device), js_input.to(device)
          
          inputs = Variable(inputs)
          js_input = Variable(js_input)
        
          outputs = model(inputs)
         

          loss = criterion(outputs, Variable(js_input))
          err = calculate_RMSE(outputs, js_input)
          
          losses.update(loss.data, inputs.size(0))
          errors.update(err, inputs.size(0))
    
          #print(js_input)
          #print(outputs)
          
          batch_time.update(time.time() - end_time)
          end_time = time.time()

          
          print('Epoch/batch/batch size: [{0}]/[{1}/{2}]\t'
              'Batch train time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data read time {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Error {err.val:.3f} ({err.avg:.3f})'.format(
                  epoch,
                  i_batch + 1,
                  len(sample_batched['inputs']),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  err=errors)) 
                    
      logger.log({'epoch': epoch, 'loss': losses.avg, 'err': errors.avg})
    return losses.avg

def test(dataloader, model, device):

    model.eval()  
    with torch.no_grad():
      batch_time = AverageMeter()
      data_time = AverageMeter()
      errors = AverageMeter()
      
      end_time = time.time()

      for i_batch, sample_batched in enumerate(dataloader):
        data_time.update(time.time() - end_time)
        
        inputs = sample_batched['inputs']
        js_input = sample_batched['js']
        inputs, js_input = inputs.to(device), js_input.to(device)
          
        inputs = Variable(inputs)
        js_input = Variable(js_input)
        
        outputs = model(inputs)
        err = calculate_RMSE(outputs, js_input)
        
        errors.update(err, inputs.size(0))
        
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Err {err.val:.3f} ({err.avg:.3f})'.format(
                  i_batch + 1,
                  len(sample_batched['inputs']),
                  batch_time=batch_time,
                  data_time=data_time,
                  err=errors))
