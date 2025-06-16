/**
 * Lock/Unlock Manager for Learning Goals Application
 * Provides password-protected editing lock functionality
 */

class LockManager {
    constructor() {
        this.password = "Modeling!";
        this.lockKey = "learning_goals_lock_state";
        this.isLocked = this.getStoredLockState();
        this.initialize();
    }

    initialize() {
        this.createLockButton();
        this.applyLockState();
        
        // Listen for storage events to sync across tabs
        window.addEventListener('storage', (e) => {
            if (e.key === this.lockKey) {
                this.isLocked = this.getStoredLockState();
                this.applyLockState();
            }
        });

        // Set up mutation observer to watch for dynamically added content
        this.setupMutationObserver();
    }

    setupMutationObserver() {
        // Create a mutation observer to watch for new elements being added
        this.observer = new MutationObserver((mutations) => {
            let shouldReapplyLock = false;
            
            mutations.forEach((mutation) => {
                if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                    // Check if any of the added nodes contain elements we need to lock
                    mutation.addedNodes.forEach((node) => {
                        if (node.nodeType === Node.ELEMENT_NODE) {
                            // Check if the added node or its descendants contain lockable elements
                            const hasLockableContent = 
                                node.querySelector && (
                                    node.querySelector('.cluster-representative') ||
                                    node.querySelector('.auto-generate-btn') ||
                                    node.querySelector('[contenteditable="true"]') ||
                                    node.querySelector('.goal-delete-btn') ||
                                    node.querySelector('.goal-move-btn') ||
                                    node.classList.contains('cluster-representative') ||
                                    node.classList.contains('auto-generate-btn') ||
                                    node.hasAttribute('contenteditable')
                                );
                            
                            if (hasLockableContent) {
                                shouldReapplyLock = true;
                            }
                        }
                    });
                }
            });
            
            // Reapply lock state if lockable content was added
            if (shouldReapplyLock && this.isLocked) {
                // Use a small delay to ensure the DOM is fully updated
                setTimeout(() => {
                    this.applyLockState();
                }, 50);
            }
        });

        // Start observing
        this.observer.observe(document.body, {
            childList: true,
            subtree: true,
            attributes: false
        });
    }

    getStoredLockState() {
        const stored = sessionStorage.getItem(this.lockKey);
        return stored === null ? true : stored === 'true'; // Default to locked
    }

    setStoredLockState(locked) {
        sessionStorage.setItem(this.lockKey, locked.toString());
    }

    createLockButton() {
        // Create the lock button container
        const lockContainer = document.createElement('div');
        lockContainer.id = 'lock-container';
        lockContainer.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 9999;
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(255, 255, 255, 0.95);
            padding: 8px 12px;
            border-radius: 20px;
            border: 1px solid #dee2e6;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(5px);
            transition: all 0.3s ease;
        `;

        // Create the lock button
        const lockButton = document.createElement('button');
        lockButton.id = 'lock-button';
        lockButton.className = 'btn btn-sm';
        lockButton.style.cssText = `
            border: none;
            background: none;
            padding: 4px 8px;
            border-radius: 15px;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 4px;
            font-size: 0.8rem;
        `;
        
        lockButton.addEventListener('click', () => this.toggleLock());

        // Create status text
        const statusText = document.createElement('span');
        statusText.id = 'lock-status-text';
        statusText.style.cssText = `
            font-size: 0.75rem;
            font-weight: 500;
            transition: color 0.2s ease;
        `;

        lockContainer.appendChild(lockButton);
        lockContainer.appendChild(statusText);
        document.body.appendChild(lockContainer);

        this.lockButton = lockButton;
        this.statusText = statusText;
        this.updateButtonAppearance();
    }

    updateButtonAppearance() {
        if (this.isLocked) {
            this.lockButton.innerHTML = '<i class="bi bi-lock-fill"></i>';
            this.lockButton.style.cssText += 'background-color: #dc3545; color: white;';
            this.lockButton.title = 'Page is locked - click to unlock';
            this.statusText.textContent = 'Locked';
            this.statusText.style.color = '#dc3545';
        } else {
            this.lockButton.innerHTML = '<i class="bi bi-unlock-fill"></i>';
            this.lockButton.style.cssText += 'background-color: #28a745; color: white;';
            this.lockButton.title = 'Page is unlocked - click to lock';
            this.statusText.textContent = 'Unlocked';
            this.statusText.style.color = '#28a745';
        }
    }

    toggleLock() {
        if (this.isLocked) {
            this.promptForPassword('unlock');
        } else {
            this.promptForPassword('lock');
        }
    }

    promptForPassword(action) {
        const message = action === 'unlock' ? 
            'Enter password to unlock editing:' : 
            'Enter password to lock editing:';
        
        const userPassword = prompt(message);
        
        if (userPassword === this.password) {
            this.isLocked = !this.isLocked;
            this.setStoredLockState(this.isLocked);
            this.applyLockState();
            this.updateButtonAppearance();
            
            const statusMessage = this.isLocked ? 'Page locked' : 'Page unlocked';
            this.showToast(statusMessage, this.isLocked ? 'danger' : 'success');
        } else if (userPassword !== null) {
            this.showToast('Incorrect password', 'danger');
        }
    }

    applyLockState() {
        if (this.isLocked) {
            this.lockPage();
        } else {
            this.unlockPage();
        }
    }

    lockPage() {
        // Disable representative text editing in artifacts page - use multiple selectors to catch all variations
        const editableSelectors = [
            '.cluster-representative[contenteditable]',
            '.cluster-representative[contenteditable="true"]',
            'div[contenteditable="true"]',
            '[data-state][contenteditable]'
        ];
        
        editableSelectors.forEach(selector => {
            document.querySelectorAll(selector).forEach(el => {
                // Only target elements that look like cluster representatives
                if (el.classList.contains('cluster-representative') || 
                    el.querySelector('.representative-text') ||
                    el.hasAttribute('data-state')) {
                    
                    el.setAttribute('data-original-contenteditable', el.getAttribute('contenteditable') || 'true');
                    el.setAttribute('contenteditable', 'false');
                    el.style.pointerEvents = 'none';
                    el.style.opacity = '0.6';
                    el.style.cursor = 'not-allowed';
                }
            });
        });

        // Disable auto-generate buttons - catch all variations
        document.querySelectorAll('.auto-generate-btn, button[onclick*="openAIGenerationModal"]').forEach(el => {
            el.disabled = true;
            el.style.pointerEvents = 'none';
            el.style.opacity = '0.4';
            el.style.cursor = 'not-allowed';
        });

        // Disable delete buttons in artifacts page
        document.querySelectorAll('.goal-delete-btn, #delete-artifact-btn, button[onclick*="deleteGoal"]').forEach(el => {
            el.disabled = true;
            el.style.pointerEvents = 'none';
            el.style.opacity = '0.4';
            el.style.cursor = 'not-allowed';
        });

        // Disable delete buttons in search page
        document.querySelectorAll('.delete-document, .batch-delete-btn').forEach(el => {
            el.disabled = true;
            el.style.pointerEvents = 'none';
            el.style.opacity = '0.4';
            el.style.cursor = 'not-allowed';
        });

        // Disable move buttons
        document.querySelectorAll('.goal-move-btn, .move-btn, button[onclick*="openMoveModal"]').forEach(el => {
            el.disabled = true;
            el.style.pointerEvents = 'none';
            el.style.opacity = '0.4';
            el.style.cursor = 'not-allowed';
        });

        // Add visual indication to body
        document.body.classList.add('page-locked');
    }

    unlockPage() {
        // Re-enable representative text editing - check all elements that might have been locked
        document.querySelectorAll('[data-original-contenteditable]').forEach(el => {
            const originalValue = el.getAttribute('data-original-contenteditable');
            el.setAttribute('contenteditable', originalValue);
            el.removeAttribute('data-original-contenteditable');
            el.style.pointerEvents = '';
            el.style.opacity = '';
            el.style.cursor = '';
        });

        // Re-enable auto-generate buttons - catch all variations
        document.querySelectorAll('.auto-generate-btn, button[onclick*="openAIGenerationModal"]').forEach(el => {
            el.disabled = false;
            el.style.pointerEvents = '';
            el.style.opacity = '';
            el.style.cursor = '';
        });

        // Re-enable delete buttons in artifacts page
        document.querySelectorAll('.goal-delete-btn, #delete-artifact-btn, button[onclick*="deleteGoal"]').forEach(el => {
            el.disabled = false;
            el.style.pointerEvents = '';
            el.style.opacity = '';
            el.style.cursor = '';
        });

        // Re-enable delete buttons in search page
        document.querySelectorAll('.delete-document, .batch-delete-btn').forEach(el => {
            el.disabled = false;
            el.style.pointerEvents = '';
            el.style.opacity = '';
            el.style.cursor = '';
        });

        // Re-enable move buttons
        document.querySelectorAll('.goal-move-btn, .move-btn, button[onclick*="openMoveModal"]').forEach(el => {
            el.disabled = false;
            el.style.pointerEvents = '';
            el.style.opacity = '';
            el.style.cursor = '';
        });

        // Remove visual indication from body
        document.body.classList.remove('page-locked');
    }

    showToast(message, type = 'info') {
        // Create toast container if it doesn't exist
        let toastContainer = document.getElementById('toast-container');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.id = 'toast-container';
            toastContainer.style.cssText = `
                position: fixed;
                top: 80px;
                right: 20px;
                z-index: 10000;
            `;
            document.body.appendChild(toastContainer);
        }

        // Create toast
        const toast = document.createElement('div');
        toast.className = `alert alert-${type} alert-dismissible`;
        toast.style.cssText = `
            margin-bottom: 10px;
            animation: slideInRight 0.3s ease;
            min-width: 200px;
        `;
        toast.innerHTML = `
            ${message}
            <button type="button" class="btn-close" onclick="this.parentElement.remove()"></button>
        `;

        toastContainer.appendChild(toast);

        // Auto-remove after 3 seconds
        setTimeout(() => {
            if (toast.parentElement) {
                toast.style.animation = 'slideOutRight 0.3s ease';
                setTimeout(() => toast.remove(), 300);
            }
        }, 3000);
    }

    // Public methods for external use
    getLockState() {
        return this.isLocked;
    }

    forceReapplyLockState() {
        // Force a comprehensive reapplication of lock state
        console.log('🔒 Forcing reapplication of lock state, current state:', this.isLocked ? 'LOCKED' : 'UNLOCKED');
        
        // Small delay to ensure DOM is ready
        setTimeout(() => {
            this.applyLockState();
            
            // Debug: count how many elements were found and locked/unlocked
            if (this.isLocked) {
                const editableCount = document.querySelectorAll('[data-original-contenteditable]').length;
                const buttonCount = document.querySelectorAll('.auto-generate-btn[disabled], button[onclick*="openAIGenerationModal"][disabled]').length;
                console.log(`🔒 Lock applied to ${editableCount} editable elements and ${buttonCount} buttons`);
            }
        }, 100);
    }

    destroy() {
        // Clean up the mutation observer and remove elements
        if (this.observer) {
            this.observer.disconnect();
        }
        
        const lockContainer = document.getElementById('lock-container');
        if (lockContainer) {
            lockContainer.remove();
        }
        
        const toastContainer = document.getElementById('toast-container');
        if (toastContainer) {
            toastContainer.remove();
        }
    }
}

// Add animations
const style = document.createElement('style');
style.textContent = `
@keyframes slideInRight {
    from { transform: translateX(100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}

@keyframes slideOutRight {
    from { transform: translateX(0); opacity: 1; }
    to { transform: translateX(100%); opacity: 0; }
}

.page-locked {
    position: relative;
}

.page-locked::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(220, 53, 69, 0.02);
    pointer-events: none;
    z-index: 1;
}
`;
document.head.appendChild(style);

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.lockManager = new LockManager();
    });
} else {
    window.lockManager = new LockManager();
} 